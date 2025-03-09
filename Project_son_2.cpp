#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <Windows.h>


double calculate_distance(double focalLength, double realHeight, double pixelHeight) {
    return (focalLength * realHeight) / pixelHeight;
}

void SendDatatoStm(const std::string& portname, uint16_t data) {

    HANDLE hSerial = CreateFileA(portname.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);

    if (hSerial == INVALID_HANDLE_VALUE) {
        std::cerr << "seri port acilmadi " << std::endl;
        return;
    }

    DCB dcbSerialParam = { 0 };
    dcbSerialParam.DCBlength = sizeof(dcbSerialParam);

    if (!GetCommState(hSerial, &dcbSerialParam)) {
        std::cerr << "mevcut port ayari alinamadi" << std::endl;
        CloseHandle(hSerial);
        return;
    }
    dcbSerialParam.BaudRate = CBR_9600;
    dcbSerialParam.ByteSize = 8;
    dcbSerialParam.StopBits = ONESTOPBIT;
    dcbSerialParam.Parity = NOPARITY;

    if (!SetCommState(hSerial, &dcbSerialParam)) {
        std::cerr << "port ayari yapilandiralamadi" << std::endl;
        CloseHandle(hSerial);
        return;
    }


    uint8_t packet[5];
    packet[0] = 0x02;
    packet[1] = (data >> 8) & 0xFF;
    packet[2] = data & 0xFF;
    packet[3] = packet[1] ^ packet[2];
    packet[4] = 0x03;
    
    DWORD bytesWritten;

    if (!WriteFile(hSerial, packet, sizeof(packet), &bytesWritten, NULL)) {
        std::cerr << "veri gonderilemedi" << std::endl;
    }
    else {
        std::cout << "veri gonderildi (" << bytesWritten << " byte):" << std::endl;
    }
    CloseHandle(hSerial);
    //data.clear();
}

void drawTrail(cv::Mat& frame, const std::vector<cv::Point>& trail, const cv::Scalar& color) {
    for (size_t i = 1;i < trail.size();++i) {
        cv::line(frame, trail[i - 1], trail[i], color, 2);
    }
}

int main() {

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    cv::VideoCapture video_cap(0);
    if (!video_cap.isOpened()) {
        std::cout << "Kamera acilamadi!" << std::endl;
        return -1;
    }
    int width = video_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "WIDTH : " << width << std::endl;
    std::cout << "HEIGHT : " << height << std::endl;

    cv::Scalar lower_blue = cv::Scalar(100, 150, 50);
    cv::Scalar upper_blue = cv::Scalar(130, 255, 255);

    cv::Scalar lower_red_1 = cv::Scalar(0, 150, 50);
    cv::Scalar upper_red_1 = cv::Scalar(10, 255, 150);
    cv::Scalar lower_red_2 = cv::Scalar(170, 150, 50);
    cv::Scalar upper_red_2 = cv::Scalar(180, 255, 150);

    int blue_centerX = width / 2;
    int blue_centerY = height / 2;

    int red_centerX = width / 2;
    int red_centerY = height / 2;

    double focalLength = 910.0;
    double realObjectHeight = 2.1;

    std::vector<cv::Point> blueTrail;
    std::vector<cv::Point> redTrail;

    std::vector<std::pair<cv::Point, double>> blueTrailWithTime;
    cv::Point predictBluePosition = cv::Point(-1, -1);

    std::vector<std::pair<cv::Point, double>> redTrailWithTime;
    cv::Point predictRedPosition = cv::Point(-1, -1);

    double errorBlueX = 0, errorBlueY = 0;
    std::vector<double> RealCenterX;
    std::vector<double> PredictCenterX;

    //stm veri gönderirken bunlarý gönder dizi göndermek sýkýntý yaratabiliyor
    double RealCenterX_kullanilan = 0;
    double PredictCenterX_kullanilan = 0;

    double lastTime = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();

    std::string port = "COM7";

    while (video_cap.isOpened()) {

        cv::Mat frame, frame_hsv;
        cv::Mat frame_blue;
        cv::Mat frame_red, mask_red1, mask_red2;

        video_cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

        cv::inRange(frame_hsv, lower_blue, upper_blue, frame_blue);

        cv::inRange(frame_hsv, lower_red_1, upper_red_1, mask_red1);
        cv::inRange(frame_hsv, lower_red_2, upper_red_2, mask_red2);
        frame_red = mask_red1 | mask_red2;

        cv::GaussianBlur(frame_blue, frame_blue, cv::Size(3, 3), cv::BORDER_DEFAULT);
        cv::GaussianBlur(frame_red, frame_red, cv::Size(3, 3), cv::BORDER_DEFAULT);

        cv::threshold(frame_blue, frame_blue, 128, 255, cv::THRESH_BINARY);
        cv::threshold(frame_red, frame_red, 128, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours_blue, contours_red;
        cv::findContours(frame_blue, contours_blue, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(frame_red, contours_red, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point>> filtered_contour_blue, filtered_contour_red;
        double min_contours_area = 100.0;
        for (const auto& contour : contours_blue) {
            double area = cv::contourArea(contour);
            if (area > min_contours_area) {
                filtered_contour_blue.push_back(contour);
            }
        }
        for (const auto& contour : contours_red) {
            double area = cv::contourArea(contour);
            if (area > min_contours_area) {
                filtered_contour_red.push_back(contour);
            }
        }
        //önemli nokta Predict yaparken ilk real konumdan sonraki real konuma gidiyor bu deðiþimden sonra 2. real konumdan 3. real konuma giderken ilk predict ti 3.real konumun predicti
        if (!filtered_contour_blue.empty()) {
            cv::Moments M = cv::moments(filtered_contour_blue[0]);
            if (M.m00 != 0) {
                int object_centerX = static_cast<int>(M.m10 / M.m00);
                int object_centerY = static_cast<int>(M.m01 / M.m00);
                blue_centerX = object_centerX;
                blue_centerY = object_centerY;
                RealCenterX.push_back(blue_centerX); //nesnenin gerçek orta konumlarýnýn tutulduðu dizi
                RealCenterX_kullanilan = blue_centerX;
                std::cout << "Real_Position.X =" << blue_centerX << "   " << "Real_Position.Y" << blue_centerY << std::endl;
                blueTrail.push_back(cv::Point(blue_centerX, blue_centerY));
                if (blueTrail.size() > 20) {
                    blueTrail.erase(blueTrail.begin());
                }
            }
            double currentTime = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
            double deltaTime = currentTime - lastTime;
            lastTime = currentTime;
            cv::Point currentPosition(blue_centerX, blue_centerY);
            if (!blueTrailWithTime.empty()) {
                cv::Point prevPosition = blueTrailWithTime.back().first;
                double prevTime = blueTrailWithTime.back().second;
                double dx = (currentPosition.x - prevPosition.x) / (currentTime - prevTime);
                double dy = (currentPosition.y - prevPosition.y) / (currentTime - prevTime);

                predictBluePosition.x = static_cast<int>(currentPosition.x + dx * deltaTime);
                predictBluePosition.y = static_cast<int>(currentPosition.y + dy * deltaTime);
                PredictCenterX.push_back(predictBluePosition.x); // predict deðerlerinin tutulduðu dizi
                PredictCenterX_kullanilan = predictBluePosition.x;

                std::cout << "Predict.X =" << predictBluePosition.x << "   " << " Predict.Y =" << predictBluePosition.y << std::endl;
                std::cout << "ErrorBlueX =" << errorBlueX << "  " << "ErrorBlueY =" << errorBlueY << std::endl;

                cv::circle(frame, predictBluePosition, 5, cv::Scalar(0, 255, 255), -1);
                cv::putText(frame, "Predict", predictBluePosition + cv::Point(10, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

            }
            blueTrailWithTime.push_back({ currentPosition,currentTime });
            if (blueTrailWithTime.size() > 20) {
                blueTrailWithTime.erase(blueTrailWithTime.begin());
            }
        }
        if (filtered_contour_blue.empty()) {
            blueTrail.clear();
            blueTrailWithTime.clear();
            predictBluePosition = cv::Point(-1, -1);
        }
        drawTrail(frame, blueTrail, cv::Scalar(255, 0, 0));
        SendDatatoStm(port,RealCenterX_kullanilan);
        //if (!filtered_contour_red.empty()) {
        //    cv::Moments MM = cv::moments(filtered_contour_red[0]);
        //    if (MM.m00 != 0) {
        //        int object_centerX = static_cast<int>(MM.m10 / MM.m00);
        //        int object_centerY = static_cast<int>(MM.m01 / MM.m00);
        //        red_centerX = object_centerX;
        //        red_centerY = object_centerY;

        //        redTrail.push_back(cv::Point(red_centerX, red_centerY));
        //        if (redTrail.size() > 20) {
        //            redTrail.erase(redTrail.begin());
        //        }
        //    }
        //    double currenTime = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
        //    double deltaTime = currenTime - lastTime;
        //    lastTime = currenTime;
        //    cv::Point currentPosition(red_centerX, red_centerY);
        //    if (!redTrailWithTime.empty()) {
        //        cv::Point prevPosition = redTrailWithTime.back().first;
        //        double prevTime = redTrailWithTime.back().second;
        //        double dx = (currentPosition.x - prevPosition.x) / (currenTime - prevTime);
        //        double dy = (currentPosition.y - prevPosition.y) / (currenTime - prevTime);

        //        predictRedPosition.x = static_cast<int>(currentPosition.x + dx * deltaTime);
        //        predictRedPosition.y = static_cast<int>(currentPosition.y + dy * deltaTime);
        //        cv::circle(frame, predictRedPosition, 5, cv::Scalar(0, 255, 255), -1);
        //        cv::putText(frame, "Predict", predictRedPosition + cv::Point(10, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        //    }
        //    redTrailWithTime.push_back({ currentPosition,currenTime });
        //    if (redTrailWithTime.size() > 20) {
        //        redTrailWithTime.erase(redTrailWithTime.begin());
        //    }
        //}
        //if (filtered_contour_red.empty()) {
        //    redTrail.clear();
        //    redTrailWithTime.clear();
        //    predictRedPosition = cv::Point(-1, -1);
        //}
        //drawTrail(frame, redTrail, cv::Scalar(0, 255, 0));

        cv::line(frame, cv::Point(0, blue_centerY), cv::Point(width, blue_centerY), cv::Scalar(120, 120, 120), 1);
        cv::line(frame, cv::Point(blue_centerX, 0), cv::Point(blue_centerX, height), cv::Scalar(120, 120, 120), 1);

        //cv::line(frame, cv::Point(0, red_centerY), cv::Point(width, red_centerY), cv::Scalar(200, 200, 200), 1);
        //cv::line(frame, cv::Point(red_centerX, 0), cv::Point(red_centerX, height), cv::Scalar(200, 200, 200), 1);

        cv::circle(frame, cv::Point(blue_centerX, blue_centerY), 5, cv::Scalar(0, 0, 255), -1);
        /*cv::circle(frame, cv::Point(red_centerX, red_centerY), 5, cv::Scalar(0, 0, 255), -1);*/

        cv::drawContours(frame, filtered_contour_blue, -1, cv::Scalar(0, 255, 0), 3);
        /*cv::drawContours(frame, filtered_contour_red, -1, cv::Scalar(0, 255, 0), 3);*/

        std::string position_text = "Center : (" + std::to_string(blue_centerX) + "," + std::to_string(blue_centerY) + ")";
        cv::putText(frame, position_text, cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        cv::imshow("Nesne Takibi", frame);

        int key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    video_cap.release();
    cv::destroyAllWindows();
    return 0;

}