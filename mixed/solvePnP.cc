/**
 * This program takes a set of corresponding 2D and 3D points and finds the transformation matrix
 * that best brings the 3D points to their corresponding 2D points.
 */
#include "solvePnP.h"
#include "utils.h"

#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    std::vector<Data3D> question = {{.1, .1, -.1}, {.2, .1, .4}};
    std::vector<Data2D> ref2D = {{282, 274}, {397, 227}, {577, 271}, {462, 318}, {270, 479}, {450, 523}, {566, 475}};
    std::vector<Data3D> ref3D = {{.5, .5, -.5}, {.5, .5, .5}, {-.5, .5, .5}, {-.5, .5, -.5}, {.5, -.5, -.5}, {-.5, -.5, -.5}, {-.5, -.5, .5}};
    std::vector<cv::Point2f> response = guessPoints(question, ref2D, ref3D);
    for (unsigned int i = 0; i < response.size(); ++i)
    {
        std::cout << "Projected to " << response[i] << std::endl;
    }

    std::string encontrado = getRegexGroup("([^/]+)$", "esto/es/una/prueba.jpg", 1);
    std::cout << encontrado << std::endl;

    return 0;
}