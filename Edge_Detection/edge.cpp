// edge
// Created by Justin Bahr on 3/23/2025.
// EECE 5640 - High Performance Computing
// Sobel Filter Edge Detection

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

using namespace std;

// Function prototype for CUDA processing
void processImageCUDA(unsigned char *h_rgbData, unsigned char *h_outputData, int width, int height);

// Function to read a PPM image
bool readPPM(const char *filename, unsigned char *&data, int &width, int &height)
{
    ifstream file(filename, ios::binary);
    if (!file)
    {
        cerr << "Error opening file: " << filename << endl;
        return false;
    }

    string header;
    file >> header;

    if (header != "P6")
    {
        cerr << "Unsupported PPM format!" << endl;
        return false;
    }

    file >> width >> height;
    int maxval;
    file >> maxval;
    file.ignore(); // Consume the newline character

    int size = width * height * 3;
    data = new unsigned char[size];
    file.read(reinterpret_cast<char *>(data), size);

    return true;
}

// Function to write a grayscale PPM image
bool writePGM(const char *filename, unsigned char *data, int width, int height)
{
    ofstream file(filename, ios::binary);
    if (!file)
    {
        cerr << "Error creating file: " << filename << endl;
        return false;
    }

    file << "P5\n" << width << " " << height << "\n255\n"; // P5 is for grayscale
    file.write(reinterpret_cast<char *>(data), width * height);
    return true;
}

int main()
{
    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the program clock
    auto program_start_time = clock::now();

    // defines the number of files to process
    const int NUM_FILES = 4;

    // creates a string array to store the input filenames
    string filenames[NUM_FILES] = {"640x426", "1280x853", "1920x1280", "5184x3456"};

    // creates string variables for the input and output file paths
    string inputPath;
    string outputPath;

    int width, height;
    unsigned char *h_rgbData;
    unsigned char *h_outputData;

    // performs the audio processing workload on all input files
    for (int fileIndex = 0; fileIndex < NUM_FILES; fileIndex++)
    {
        cout << "Processing image size: " << filenames[fileIndex] << endl;

        // starts an image clock
        auto start_time = clock::now();

        // saves the input and output file paths
        inputPath = "Input_Samples/";
        outputPath = "Output_Samples/";

        inputPath += filenames[fileIndex] + ".ppm";
        outputPath += filenames[fileIndex] + "_out.pgm";

        // creates points to store the input and output file path arguments
        const char* inputArg = inputPath.c_str();
        const char* outputArg = outputPath.c_str();

        // Read the input PPM image
        if (!readPPM(inputArg, h_rgbData, width, height))
        {
            return EXIT_FAILURE;
        }

        // stores the size of the gray image version
        size_t graySize = width * height;

        // Allocate memory on host and device
        h_outputData = new unsigned char[graySize];

        // Process image with CUDA
        processImageCUDA(h_rgbData, h_outputData, width, height);

        // Write output PPM image
        if (!writePGM(outputArg, h_outputData, width, height))
        {
            return EXIT_FAILURE;
        }

        // stops an image clock
        auto end_time = clock::now();

        // casts image_run_time in nanoseconds
        auto image_run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

        cout << "Time to process in nanoseconds: " << image_run_time << endl << endl;
    }

    // free allocated memory
    delete[] h_rgbData;
    delete[] h_outputData;
    h_rgbData = nullptr;
    h_outputData = nullptr;

    // stops the program clock
    auto program_end_time = clock::now();

    // casts program_run_time in nanoseconds
    auto program_run_time = chrono::duration_cast<chrono::nanoseconds>(program_end_time - program_start_time).count();

    cout << "Total program runtime in nanoseconds: " << program_run_time << endl;

    return 0;
}