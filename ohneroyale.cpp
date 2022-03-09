/****************************************************************************\
* Vorlage fuer das Praktikum "Bildverarbeitung" WS 2021/22
* FB 03 der Hochschule Niederrhein
* Christian Neumann, Regina Pohle-Froehlich
*
* Der Code basiert auf den C++-Beispielen der Bibliothek royale
\****************************************************************************/

#include <mutex>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <chrono>
#include <thread>

using namespace std;
using namespace cv;

// Funktion zur Berechnung des Schwellwertes
void schwellwert(Mat &bild)
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
    double ssteigung = 0;

    for (int i = 10; i < 256; i++)
    {
        double steigung = ((glatt[i + 1] + glatt[i + 2] + glatt[i + 3]) / 3) - glatt[i];
        ssteigung = steigung;
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
    for (int i = 0; i < approxCurve.size(); ++i)
    {
        for (const Vec4i &v : defects[i])
        {
            float depth = v[3] / 256;
            if (depth > 5) //  filter defects by depth, e.g more than 5
            {
                int startidx = v[0];
                Point ptStart(approxCurve[i][startidx]);
                int endidx = v[1];
                Point ptEnd(approxCurve[i][endidx]);
                int faridx = v[2];
                Point ptFar(approxCurve[i][faridx]);

                line(kontur, ptStart, ptEnd, Scalar(0, 255, 0), 1);
                line(kontur, ptStart, ptFar, Scalar(0, 255, 0), 1);
                line(kontur, ptEnd, ptFar, Scalar(0, 255, 0), 1);
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

    waitKey(1);
}

int main(int argc, char *argv[])
{
    string filename;

    filename = "C:\\Users\\ibo_s\\Desktop\\P2v3\\P2\\aa.avi";
    VideoCapture cap = VideoCapture(filename);
    if (!cap.isOpened())
    {
        cout << "Konnte Video nicht �ffnen!" << endl;
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

        // cvtColor(frame, frame, COLOR_BGR2GRAY, 0);
        if (!frame.empty())
        {
            Mat test;
            cvtColor(frame, test, COLOR_BGR2GRAY);
            schwellwert(test);

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

    return 0;
}