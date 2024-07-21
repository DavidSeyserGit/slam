#include <ios>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

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
    cv::findHomography(pts1, pts2, cv::RANSAC, 10, mask);

    std::vector<cv::DMatch> inliers;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (mask[i]) {
            inliers.push_back(matches[i]);
        }
    }
    return inliers;
}

int main()
{
    cv::Mat frame, desc, desc2, frame2;
    cv::VideoCapture cap;
    std::vector<cv::KeyPoint> kp, kp2;
    std::vector<cv::DMatch> matches;


    // max 1215752192
    //     1000000
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000000);
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    if (!cap.open(0)) {
        std::cerr << "Error opening camera" << std::endl;
        return -1;
    }

    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "Error capturing frame" << std::endl;
            break;
        }

        // Downsample the frame for faster processing
        cv::Mat smallFrame;
        cv::resize(frame, smallFrame, cv::Size(), 0.5, 0.5);

        orb->detectAndCompute(smallFrame, cv::noArray(), kp, desc);

        cv::Mat blackBackground = cv::Mat::zeros(frame.size(), frame.type());

        // Match descriptors if there are previous descriptors
        if (!desc2.empty()) {
            matcher.match(desc, desc2, matches);
            std::vector<cv::DMatch> inliers = filterMatches(kp, kp2, matches);

            std::sort(inliers.begin(), inliers.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance;
            });

            const size_t maxMatches = 50;
            for (size_t i = 0; i < std::min(inliers.size(), maxMatches); ++i) {
                const auto& match = inliers[i];
                // Scale keypoints back to the original frame size
                cv::Point2f pt1 = kp[match.queryIdx].pt * 2.0f;
                cv::Point2f pt2 = kp2[match.trainIdx].pt * 2.0f;
                cv::line(blackBackground, pt2, pt1, cv::Scalar(0, 0, 255), 1);
            }
        }

        std::vector<cv::KeyPoint> scaledKp;
        for (const auto& keypoint : kp) {
            scaledKp.push_back(cv::KeyPoint(keypoint.pt * 2.0f, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id));
        }

        cv::Mat out;
        cv::drawKeypoints(blackBackground, scaledKp, out, cv::Scalar(255, 200, 50), cv::DrawMatchesFlags::DEFAULT);

        cv::imshow("frame", out);

        if (cv::waitKey(30) >= 0) {
            break;
        }

        // Properly assign new values to frame2, desc2, and kp2
        frame2 = frame.clone();
        desc2 = desc.clone();
        kp2 = kp;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
