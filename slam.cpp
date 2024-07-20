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
    cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);

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
    cv::Mat frame, desc;
    cv::Mat frame2, desc2;
    cv::VideoCapture cap;

    std::vector<cv::KeyPoint> kp, kp2;
    std::vector<cv::DMatch> matches;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(50000);
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    if(!cap.open(0)){
        std::cerr << "Error opening camera" << std::endl;
        return -1;
    }

    while (true) {
        cap.read(frame);

        if (frame.empty()) {
            std::cerr << "Error capturing frame" << std::endl;
            break;
        }

        // Clear previous keypoints
        //kp.clear();

        // Detect and compute features
        orb->detectAndCompute(frame, cv::noArray(), kp, desc);

        // Create a black background image of the same size as the frame
        cv::Mat blackBackground = cv::Mat::zeros(frame.size(), frame.type());

        // Match descriptors if there are previous descriptors
        if (!desc2.empty()) {
            matcher.match(desc, desc2, matches);
            std::vector<cv::DMatch> inliers = filterMatches(kp, kp2, matches);

            // Draw matches
            for (size_t i = 0; i < std::min(inliers.size(), size_t(100)); ++i) {
                const auto& match = inliers[i];
                cv::line(blackBackground, kp2[match.trainIdx].pt, kp[match.queryIdx].pt, cv::Scalar(0, 0, 255), 2);
            }
        }

        // Draw keypoints on the black background
        cv::Mat out;
        cv::drawKeypoints(blackBackground, kp, out, cv::Scalar(255, 200, 50), cv::DrawMatchesFlags::DEFAULT);

        cv::imshow("frame", out);

        if (cv::waitKey(30) >= 0){
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
