/****************************************************************************\
* Vorlage fuer das Praktikum "Bildverarbeitung" WS 2021/22
* FB 03 der Hochschule Niederrhein
* Christian Neumann, Regina Pohle-Froehlich
*
* Der Code basiert auf den C++-Beispielen der Bibliothek royale
\****************************************************************************/

#include <mutex>
#include <iostream>
#include <royale.hpp>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <chrono>
#include <thread>

using namespace std;
using namespace cv;

class MyListener : public royale::IDepthDataListener
{
public:
    void onNewData(const royale::DepthData *data)
    {

        // this callback function will be called for every new depth frame
        has_data = false;
        cv::Mat zImage(data->height, data->width, CV_32FC1, cv::Scalar(0.0f));
        int k = 0;
        for (int y = 0; y < zImage.rows; y++)
        {
            for (int x = 0; x < zImage.cols; x++)
            {
                const royale::DepthPoint curPoint = data->points.at(k);
                if (curPoint.depthConfidence > 80 && curPoint.z < 3.5f)
                {
                    // if the point is valid
                    zImage.at<float>(y, x) = curPoint.z;
                }
                k++;
            }
        }
        {
            std::lock_guard<std::mutex> lck(mtx);
            undistort(zImage, MyListener::zImage, cameraMatrix, distortionCoefficients);
        }
        has_data = true;
    }

    void setLensParameters(const royale::LensParameters &lensParameters)
    {
        // Construct the camera matrix
        // (fx   0    cx)
        // (0    fy   cy)
        // (0    0    1 )
        cameraMatrix = (cv::Mat1d(3, 3) << lensParameters.focalLength.first, 0.0f, lensParameters.principalPoint.first,
                        0.0f, lensParameters.focalLength.second, lensParameters.principalPoint.second,
                        0.0f, 0.0f, 1.0f);

        // Construct the distortion coefficients
        // k1 k2 p1 p2 k3
        distortionCoefficients = (cv::Mat1d(1, 5) << lensParameters.distortionRadial[0],
                                  lensParameters.distortionRadial[1],
                                  lensParameters.distortionTangential.first,
                                  lensParameters.distortionTangential.second,
                                  lensParameters.distortionRadial[2]);
    }

    bool hasData() const
    {
        return has_data;
    }

    void showImage()
    {

        Point minLoc, maxLoc;
        double min, max;
        Mat mask;

        compare(zImage, 0, mask, CMP_NE);
        minMaxLoc(zImage, &min, &max, &minLoc, &maxLoc, mask);

        double alpha = 255 / (max - min);
        double beta = -min * alpha;

        convertScaleAbs(zImage, zImage, alpha, beta);
        //applyColorMap(zImage, colorImage, COLORMAP_RAINBOW);

        imshow("Live_Video", zImage);

        waitKey(1);
    }

    cv::Mat zImage;
    std::mutex mtx;
    bool has_data = false;

private:
    // lens matrices used for the undistortion of the image
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;
};

//ConvexHull Funktion
vector<Point> contoursConvexHull( vector<vector<Point> > contours )
{
    vector<Point> result;
    vector<Point> pts;
    for ( size_t i = 0; i< contours.size(); i++)
        for ( size_t j = 0; j< contours[i].size(); j++)
            pts.push_back(contours[i][j]);
    convexHull( pts, result );
    return result;
}



// Funktion zur Berechnung des Schwellwertes
void schwellwert(Mat &bild, int tiefe)
{

    int histogramm[256]{};
    int glatt[256]{};
    int x, y;
    x = bild.rows;
    y = bild.cols;
    Mat binaer = bild.clone();
    Mat kontur = bild.clone();

    // Histo mit Werten des Bildes f�llen
    for (int i = 0; i < x; ++i)
    {

        for (int j = 0; j < y; ++j)
        {

            ++histogramm[bild.at<uchar>(i, j)];
        }
    }

    int hist_summe = 0;
    int hist_anzahl = 0;
    for (int i = 0; i < 256; i++)
    {
        int count = 0;
        int sum = 0;
        hist_summe += histogramm[i];
        hist_anzahl++;
        for (int j = i - 10; j <= i + 10; j++)
        {

            if (j >= 0 && j <= 255)
            {

                count++;
                sum += histogramm[j];
            }
        }

        glatt[i] = sum / count;
    }

    int max2 = 0;

    // Max Wert im Histogrammm Glatt finden
    for (int i = 0; i < 256; ++i)
    {

        if (max2 <= glatt[i])
        {
            max2 = glatt[i];
        }
        // cout<<"Hist: "<<histogramm[i]<<endl;
    }

    // steigung
    double steigung_alt = 0;
    int maximum_x = 0;
    int minimum_x = 0;

    for (int i = 10; i < 256; i++)
    {
        double steigung = ((glatt[i + 1] + glatt[i + 2] + glatt[i + 3]) / 3) - glatt[i];
        if (maximum_x == 0)
        {
            if (steigung_alt > 0 && steigung <= 0)
            {
                maximum_x = i;
            }
        }
        else if (minimum_x == 0)
        {
            if (steigung_alt < 0 && steigung >= 0)
            {
                minimum_x = i;
            }
        }
        else
        {
            break;
        }
        steigung_alt = steigung;
    }

    // histo variablen
    int thick = 1;
    int start = 0;
    int end = 0;
    int hoehe = 500;
    Mat hist_img(hoehe, 256, CV_32F, Scalar(0));

    // histo zeichnen
    for (int i = 0; i < 256 - 1; ++i)
    {

        start = (glatt[i] * hoehe) / max2;
        end = (glatt[i + 1] * hoehe) / max2;

        line(hist_img, Point(i, hoehe - start), Point(i + 1, hoehe - end), Scalar(255), thick, LINE_8);
    }

    line(hist_img, Point((int)(hoehe / 255) * maximum_x, 0), Point((int)(hoehe / 255) * maximum_x, hoehe), Scalar(255), 2, LINE_8);
    line(hist_img, Point((int)(hoehe / 255) * minimum_x, 0), Point((int)(hoehe / 255) * minimum_x, hoehe), Scalar(255), 1, LINE_8);

    for (int i = 0; i < x; ++i)
    {

        for (int j = 0; j < y; ++j)
        {

            if (binaer.at<uchar>(i, j) > 0 && binaer.at<uchar>(i, j) < minimum_x)
                binaer.at<uchar>(i, j) = 255;
            else
                binaer.at<uchar>(i, j) = 0;
        }
    }

    vector<vector<Point>> countours;
    std::vector<Vec4i> hierarchy;
    vector<vector<Point>> approxCurve;

    morphologyEx(binaer, binaer, MORPH_OPEN, getStructuringElement(MORPH_CROSS, Size(4, 4)), Point(-1, -1), 2);
    imshow("Binary Mat", binaer);

    findContours(binaer, countours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE); // vllt. CHAIN_APPROX_SIMPLE

    // Douglas Algo Kontur Poly
    approxCurve.resize(countours.size());
    for (size_t k = 0; k < countours.size(); k++)
        approxPolyDP(Mat(countours[k]), approxCurve[k], 2, true);

    vector<vector<Point>> hull(approxCurve.size()); // das selbe wie hull.resize(approxCurve.size())
    vector<vector<int>> hullsI(approxCurve.size()); // Indices to contour points
    vector<vector<Vec4i>> defects(approxCurve.size());

    for (size_t i = 0; i < approxCurve.size(); i++)
    {
        convexHull(approxCurve[i], hull[i], false);
        convexHull(approxCurve[i], hullsI[i], false);

        if (hullsI[i].size() > 3) // You need more than 3 indices
        {
            convexityDefects(approxCurve[i], hullsI[i], defects[i]);
        }
    }

    Mat result;
    cvtColor(kontur, result, COLOR_GRAY2BGR);

    drawContours(result, approxCurve, -1, Scalar(255, 255, 255), 1);
    drawContours(result, hull, -1, Scalar(0, 255, 0), 1);

    // draw defects
    int count = 0;
    for ( int i = 0; i < approxCurve.size(); ++i)
    {
        for (const Vec4i &v : defects[i])
        {
            float depth = v[3] / 256;
            if (depth > tiefe)
            {
                int startidx = v[0];
                Point ptStart(approxCurve[i][startidx]);
                int endidx = v[1];
                Point ptEnd(approxCurve[i][endidx]);
                int faridx = v[2];
                Point ptFar(approxCurve[i][faridx]);

                /* line(result, ptStart, ptEnd, Scalar(0, 255, 0), 1);
                line(result, ptStart, ptFar, Scalar(0, 255, 0), 1);
                line(result, ptEnd, ptFar, Scalar(0, 255, 0), 1); */
                circle(result, ptFar, 4, Scalar(0, 242, 255), 2);
                count++;
            }
        }
    }

    if (count > 1 && count < 5)
    {
        cv::putText(result,               // target image
                    to_string(count - 1), // text
                    cv::Point(10, 15),    // top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    0.5,
                    CV_RGB(255, 0, 0), // font color
                    2);
    }

    imshow("Histogram", hist_img);
    imshow("Contour Mat", result);

    //waitKey(32);
}

int main(int argc, char *argv[])
{

    MyListener listener;

    // this represents the main camera device object
    std::unique_ptr<royale::ICameraDevice> cameraDevice;

    // the camera manager will query for a connected camera
    {
        royale::CameraManager manager;

        // try to open the first connected camera
        royale::Vector<royale::String> camlist(manager.getConnectedCameraList());
        std::cout << "Detected " << camlist.size() << " camera(s)." << std::endl;

        if (!camlist.empty())
        {
            cameraDevice = manager.createCamera(camlist[0]);
        }
        else
        {
            std::cerr << "No suitable camera device detected." << std::endl
                      << "Please make sure that a supported camera is plugged in, all drivers are " << std::endl
                      << "installed, and you have proper USB permission" << std::endl;
            return 1;
        }

        camlist.clear();
    }
    // the camera device is now available and CameraManager can be deallocated here

    if (cameraDevice == nullptr)
    {
        // no cameraDevice available
        if (argc > 1)
        {
            std::cerr << "Could not open " << argv[1] << std::endl;
            return 1;
        }
        else
        {
            std::cerr << "Cannot create the camera device" << std::endl;
            return 1;
        }
    }

    // call the initialize method before working with the camera device
    royale::CameraStatus status = cameraDevice->initialize();
    if (status != royale::CameraStatus::SUCCESS)
    {
        std::cerr << "Cannot initialize the camera device, error string : " << getErrorString(status) << std::endl;
        return 1;
    }

    // retrieve the lens parameters from Royale
    royale::LensParameters lensParameters;
    status = cameraDevice->getLensParameters(lensParameters);
    if (status != royale::CameraStatus::SUCCESS)
    {
        std::cerr << "Can't read out the lens parameters" << std::endl;
        return 1;
    }

    listener.setLensParameters(lensParameters);

    // register a data listener
    if (cameraDevice->registerDataListener(&listener) != royale::CameraStatus::SUCCESS)
    {
        std::cerr << "Error registering data listener" << std::endl;
        return 1;
    }

    // start capture mode
    if (cameraDevice->startCapture() != royale::CameraStatus::SUCCESS)
    {
        std::cerr << "Error starting the capturing" << std::endl;
        return 1;
    }

    // UNSER CODE

    uint16_t fps;
    uint16_t h;
    uint16_t w;
    cameraDevice->getMaxFrameRate(fps);
    cameraDevice->getMaxSensorHeight(h);
    cameraDevice->getMaxSensorWidth(w);
    // int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');

    if ((argc == 2) && (*argv[1] == '1'))
    {
        string windowName = "RAW";
        namedWindow(windowName);

        while (true)
        {
            int key = 0;
            if (listener.hasData()) 
            {
                double min = 0;
                double max = 0;
                Point minLoc, maxLoc;
                Mat mask;

                std::lock_guard<std::mutex> lck(listener.mtx);
                compare(listener.zImage, 0, mask, CMP_NE);
                minMaxLoc(listener.zImage, &min, &max, &minLoc, &maxLoc, mask);
                convertScaleAbs(listener.zImage, listener.zImage, 255 / (max - min), -min * 255 / (max - min));
                schwellwert(listener.zImage, 10);
                listener.has_data = false;
                cv::imshow(windowName, listener.zImage);
                //key = waitKey(1);
            }
            if (waitKey(1) == 32)
                break;
        }
    }

    if ((*argv[1] == '2'))
    {
        //cout << "Aufruf der Auswertung" << endl;
        string windowName = "Window";
        namedWindow(windowName);
        Mat ucharMat;
        Mat mask;
        double min = 0;
        double max = 0;
        Size size = Size(w, h);
        string praefix = "";
        string name;

        if (argc == 2)
        {
            cout << "Dateinamen eingeben: ";
            cin >> name;
        }
        else
        {
            praefix = argv[2];
        }

        VideoWriter writer = VideoWriter("./" + praefix + name + ".avi", 0, fps, size, false);
        cameraDevice->setExposureMode(royale::ExposureMode::AUTOMATIC);
        while (true)
        {
            int key = 0;
            if (listener.hasData())
            {
                std::lock_guard<std::mutex> lck(listener.mtx);
                compare(listener.zImage, 0, mask, CMP_NE);
                minMaxLoc(listener.zImage, &max, &min, 0, 0, mask);
                listener.zImage.convertTo(ucharMat, CV_8U, 255 / (max - min), -min * 255 / (max - min));
                //applyColorMap(ucharMat, ucharMat, COLORMAP_RAINBOW);
                cv::imshow(windowName, ucharMat);
                writer.write(ucharMat);
                key = waitKey(1);
            }
            if (key > 0)
                break;
        }
        destroyWindow(windowName);
    }

    if ((*argv[1] == '3'))
    {

        string praefix;
        string filename;
        if (argc == 2)
        {
            cout << "Dateinamen des Videos eingeben: " << endl;

            cin >> filename;
        }
        else
        {
            praefix = argv[2];
        }

        //filename += ".avi";
        filename = "./" + praefix + filename + ".avi";
        VideoCapture cap = VideoCapture(filename);
        if (!cap.isOpened())
        {
            cout << "Konnte Video nicht öffnen!" << endl;
            return -1;
        }
        Mat frame;

        string windowName = "Window";
        namedWindow(windowName);

        while (true)
        {
            cap >> frame;

            // listener.has_data = false;
            int key = 0;

            //cvtColor(frame, frame, COLOR_BGR2GRAY, 0);
            if (!frame.empty())
            {
                Mat test;
                cvtColor(frame, test, COLOR_BGR2GRAY);
                schwellwert(test, 5);

                cv::imshow(windowName, frame);
                std::this_thread::sleep_for(std::chrono::milliseconds(200));

                key = waitKey(1);
            }
            else
            {
                key = waitKey(0);
                if (key > 0)
                    break;
            }
            if (key > 0)
                break;
        }
        waitKey(0);
        destroyWindow(windowName);
    }

    // stop capture mode
    if (cameraDevice->stopCapture() != royale::CameraStatus::SUCCESS)
    {
        std::cerr << "Error stopping the capturing" << std::endl;
        return 1;
    }

    return 0;
}