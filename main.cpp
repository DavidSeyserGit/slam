#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>

class CameraException : public std::runtime_error {
public:
    CameraException(const std::string& message) : std::runtime_error(message) {}
};

struct Landmark {
    cv::Point3f position;
    int seen_count;
};

std::vector<cv::DMatch> matchPoints(const cv::Mat& des1, const cv::Mat& des2) {
    cv::BFMatcher bf(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    bf.match(des1, des2, matches);
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
    });
    return matches;
}

std::vector<cv::DMatch> filterMatches(const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2, const std::vector<cv::DMatch>& matches) {
    if (matches.size() < 4) {
        return {};
    }

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }

    std::vector<uchar> mask;
    cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);

    std::vector<cv::DMatch> inliers;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (mask[i]) {
            inliers.push_back(matches[i]);
        }
    }
    return inliers;
}

cv::Mat estimatePose(const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2, const std::vector<cv::DMatch>& inliers) {
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : inliers) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) << 700, 0, 320, 0, 700, 240, 0, 0, 1);
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K);
    cv::Mat R, t;
    cv::recoverPose(E, pts1, pts2, K, R, t);

    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
    t.copyTo(pose(cv::Rect(3, 0, 1, 3)));

    return pose;
}

void drawLandmarks(cv::Mat& frame, const std::vector<Landmark>& landmarks) {
    for (const auto& landmark : landmarks) {
        cv::circle(frame, cv::Point(landmark.position.x, landmark.position.y), 3, cv::Scalar(0, 255, 0), -1);
    }
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        throw CameraException("Camera error");
    }

    cv::Ptr<cv::ORB> orb = cv::ORB::create(100000);

    cv::Mat prev_frame;
    if (!cap.read(prev_frame)) {
        throw CameraException("Could not read frame");
    }

    std::vector<cv::KeyPoint> prev_kp;
    cv::Mat prev_des;
    orb->detectAndCompute(prev_frame, cv::noArray(), prev_kp, prev_des);

    std::vector<Landmark> landmarks(prev_kp.size());
    for (size_t i = 0; i < prev_kp.size(); ++i) {
        landmarks[i] = {cv::Point3f(prev_kp[i].pt.x, prev_kp[i].pt.y, 0), 1};
    }

    cv::Mat cumulative_pose = cv::Mat::eye(4, 4, CV_64F);

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "Error reading frame" << std::endl;
            break;
        }

        std::vector<cv::KeyPoint> kp;
        cv::Mat des;
        orb->detectAndCompute(frame, cv::noArray(), kp, des);

        if (!prev_des.empty() && !des.empty()) {
            std::vector<cv::DMatch> matches = matchPoints(prev_des, des);
            std::vector<cv::DMatch> inliers = filterMatches(prev_kp, kp, matches);

            if (inliers.size() >= 5) {
                cv::Mat relative_pose = estimatePose(prev_kp, kp, inliers);
                cumulative_pose = cumulative_pose * relative_pose;
                std::cout << "Cumulative Pose: \n" << cumulative_pose << std::endl;

                for (const auto& inlier : inliers) {
                    if (inlier.queryIdx < landmarks.size()) {
                        cv::KeyPoint kp1 = prev_kp[inlier.queryIdx];
                        cv::KeyPoint kp2 = kp[inlier.trainIdx];
                        float depth = 1.0f; // Simplified; in practice, obtain depth from stereo/other sensors
                        landmarks[inlier.queryIdx].position = cv::Point3f(kp2.pt.x, kp2.pt.y, depth);
                        landmarks[inlier.queryIdx].seen_count++;
                    } else {
                        std::cerr << "Index out of bounds for landmarks vector" << std::endl;
                    }
                }
            }

            cv::Mat frame_with_keypoints;
            cv::drawKeypoints(frame, kp, frame_with_keypoints, cv::Scalar(100, 0, 0), cv::DrawMatchesFlags::DEFAULT);

            for (size_t i = 0; i < std::min(inliers.size(), size_t(100)); ++i) {
                const auto& match = inliers[i];
                cv::line(frame_with_keypoints, prev_kp[match.queryIdx].pt, kp[match.trainIdx].pt, cv::Scalar(255, 255, 255), 4);
            }

            drawLandmarks(frame_with_keypoints, landmarks);
            cv::imshow("Matches and Keypoints", frame_with_keypoints);
        } else {
            std::cerr << "Descriptors are empty" << std::endl;
        }

        if (cv::waitKey(1) == 'q') {
            break;
        }

        prev_frame = frame.clone();
        prev_kp = kp;
        prev_des = des;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
