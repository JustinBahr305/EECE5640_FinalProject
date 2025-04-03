// fir
// Created by Justin Bahr on 3/14/2025.
// EECE 5640 - High Performance Computing
// High Order Low Pass FIR Filter - Serial

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

using namespace std;

// structure to store the WAV file header
typedef struct  WAV_HEADER
{
    /* RIFF Chunk Descriptor */
    uint8_t         RIFF[4];        // RIFF Header Magic header
    uint32_t        ChunkSize;      // RIFF Chunk Size
    uint8_t         WAVE[4];        // WAVE Header
    /* "fmt" sub-chunk */
    uint8_t         fmt[4];         // FMT header
    uint32_t        Subchunk1Size;  // Size of the fmt chunk
    uint16_t        AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
    uint16_t        NumOfChan;      // Number of channels 1=Mono 2=Stereo
    uint32_t        SamplesPerSec;  // Sampling Frequency in Hz
    uint32_t        bytesPerSec;    // bytes per second
    uint16_t        blockAlign;     // 2=16-bit mono, 4=16-bit stereo
    uint16_t        bitsPerSample;  // Number of bits per sample
    /* "data" sub-chunk */
    uint8_t         Subchunk2ID[4]; // "data"  string
    uint32_t        Subchunk2Size;  // Sampled data length
} wav_hdr;

// function to find the file size
int getFileSize(FILE *inFile)
{
    int fileSize = 0;
    fseek(inFile,0,SEEK_END);

    fileSize=ftell(inFile);

    fseek(inFile,0,SEEK_SET);
    return fileSize;
}

// function to print the WAV header
void printWAVHeader(WAV_HEADER &wavHeader, int filelength)
{
    // display file information
    cout << "File is: " << filelength << " bytes." << endl;
    cout << "RIFF header                :" << wavHeader.RIFF[0] << wavHeader.RIFF[1] << wavHeader.RIFF[2] <<
        wavHeader.RIFF[3] << endl;
    cout << "WAVE header                :" << wavHeader.WAVE[0] << wavHeader.WAVE[1] << wavHeader.WAVE[2] <<
        wavHeader.WAVE[3] << endl;
    cout << "FMT                        :" << wavHeader.fmt[0] << wavHeader.fmt[1] << wavHeader.fmt[2] <<
        wavHeader.fmt[3] << endl;
    cout << "Data size                  :" << wavHeader.ChunkSize << endl << endl;

    // Display the sampling Rate form the header
    cout << "Sampling Rate              :" << wavHeader.SamplesPerSec << endl;
    cout << "Number of bits used        :" << wavHeader.bitsPerSample << endl;
    cout << "Number of channels         :" << wavHeader.NumOfChan << endl;
    cout << "Number of bytes per second :" << wavHeader.bytesPerSec << endl;
    cout << "Data length                :" << wavHeader.Subchunk2Size << endl;
    cout << "Audio Format               :" << wavHeader.AudioFormat << endl;
    cout << "Block align                :" << wavHeader.blockAlign << endl;
    cout << "Data string                :" << wavHeader.Subchunk2ID[0] << wavHeader.Subchunk2ID[1] <<
        wavHeader.Subchunk2ID[2] << wavHeader.Subchunk2ID[3] << endl << endl;
}

// function to perform FIR filtration
void FIR_lowpass(const int16_t inputL[], int16_t outputL[], const int16_t inputR[], int16_t outputR[],
    int signalLength,const float coefficients[], int order)
{
    // filters left channel
    for (int i = signalLength-1; i >= 0; i--)
    {
        float temp = 0.0;
        for (int j = 0; j < min(order,i+1); j++)
        {
            temp += coefficients[j] * inputL[i-j];
        }
        outputL[i] = (int16_t)(temp);
    }

    // filters right channel
    for (int i = signalLength-1; i >= 0; i--)
    {
        float temp = 0.0;
        for (int j = 0; j < min(order,i+1); j++)
        {
            temp += coefficients[j] * inputR[i-j];
        }
        outputR[i] = (int16_t)(temp);
    }
}

int main()
{
	cout << endl << "Running fir ..." << endl << endl;

    // initialize the high resolution clock
    typedef chrono::high_resolution_clock clock;;

    // starts the program clock
    auto program_start_time = clock::now();

    // defines the FIR filter order
    const int ORDER = 201;

    // creates a double array to store filter coefficients
    float coefficients[ORDER];

    // opens the coefficients text file
    ifstream coFile("Filters/coefficients.txt");

    // throws an error if the coefficients text file cannot be opened
    if (!coFile)
    {
        cerr << "Could not open file" << endl;
        return -1;
    }

    // reads double precision floating-point numbers line by line
    for (int i = 0; i < ORDER; ++i)
    {
        double temp;  // read as double first to preserve precision
        if (!(coFile >> temp))
        {
            cerr << "Error reading coefficient at index " << i << endl;
            return -1;
        }
        coefficients[i] = static_cast<float>(temp);  // convert to float
    }

    // closes the coefficients text file
    coFile.close();

    // defines the number of files to process
    const int NUM_FILES = 7;

    // creates a string array to store the input filenames
    string filenames[NUM_FILES] = {"StarWars4", "StarWars6", "StarWars10", "StarWars12",
        "StarWars13", "StarWars20", "StarWars29"};

    // creates string variables for the input and output file paths
    string inputPath;
    string outputPath;

    // creates a wav header
    wav_hdr wavHeader;

    // creates a file to read and write into
    FILE *wavFile;

    // creates variables to hold the header size and file length
    int headerSize = sizeof(wav_hdr);
    int filelength;

    // performs the audio processing workload on all input files
    for (int fileIndex = 0; fileIndex < NUM_FILES; fileIndex++)
    {
        cout << "Filtering file: " << filenames[fileIndex] << endl;

        // starts a read clock
        auto start_time = clock::now();

        // saves the input and output file paths
        inputPath = "Input_Samples/";
        outputPath = "Output_Samples/";

        inputPath += filenames[fileIndex] + ".wav";
        outputPath += filenames[fileIndex] + "_out.wav";

        // creates points to store the input and output file path arguments
        const char* inputArg = inputPath.c_str();
        const char* outputArg = outputPath.c_str();

        // opens the input WAV file
        wavFile = fopen(inputArg,"rb");

        // throws an error if the WAV file cannot be opened
        if (wavFile == NULL)
        {
            cerr << "Could not open file" << endl;
            return -1;
        }

        // reads the WAV header and saves the file length
        fread(&wavHeader,headerSize,1,wavFile);
        filelength = getFileSize(wavFile);

        // throws an error if the WAV file is not stored in stereo format
        if (wavHeader.NumOfChan != 2)
        {
            fclose(wavFile);
            cerr << "File is not in stereo format" << endl;
            return -1;
        }

        // creates a point for input/output data
        int16_t* data;

        // creates array pointers for the input and output stereo data
        int16_t *inputL;
        int16_t *inputR;
        int16_t *outputL;
        int16_t *outputR;

        // moves the file fread offset to bypass the header information
        fseek(wavFile, headerSize, SEEK_SET);

        // creates and fills the input data array
        int inputLength = wavHeader.Subchunk2Size / 2;
        data = new int16_t[inputLength]; // Assuming 16-bit samples
        fread(data, wavHeader.Subchunk2Size, 1,wavFile);

        // closes the input WAV file
        fclose(wavFile);

        // halves the input length due to stereo format
        inputLength /= 2;

        // allocates memory for the input and output stereo array
        inputL = new int16_t[inputLength];
        inputR = new int16_t[inputLength];;
        outputL = new int16_t[inputLength];
        outputR = new int16_t[inputLength];

        // fills the input stereo data
        for (int i = 0; i < inputLength; i++)
        {
            inputL[i] = data[2*i];
            inputR[i] = data[2*i + 1];
        }

        // stops a read clock
        auto end_time = clock::now();

        // casts read_run_time in nanoseconds
        auto read_run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

        cout << "Time to read in nanoseconds: " << read_run_time << endl;

        // starts a FIR clock
        start_time = clock::now();

        // performs FIR filtration on the input stereo data and stores in output stereo arrays
        FIR_lowpass(inputL, outputL, inputR, outputR, inputLength, coefficients, ORDER);

        // stops a FIR clock
        end_time = clock::now();

        // casts FIR_run_time in nanoseconds
        auto FIR_run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

        cout << "Time to filter in nanoseconds: " << FIR_run_time << endl;

        // starts a write clock
        start_time = clock::now();

        // fills the output data array
        for (int i = 0; i < inputLength; i++)
        {
            data[2*i] = outputL[i];
            data[2*i+1] = outputR[i];
        }

        // opens the output WAV file
        wavFile = fopen(outputArg, "wb");

        // throws an error if the WAV file cannot be opened
        if (wavFile == NULL)
        {
            cerr << "Could not open file" << endl;

            // free allocated memory
            delete[] inputL;
            delete[] inputR;
            delete[] outputL;
            delete[] outputR;
            delete[] data;
            inputL = nullptr;
            inputR = nullptr;
            outputL = nullptr;
            outputR = nullptr;
            data = nullptr;

            return -1;
        }

        // writes the WAV header into the output WAV file
        fwrite(&wavHeader,headerSize,1,wavFile);

        // moves the file fread offset to bypass the header information
        fseek(wavFile, headerSize, SEEK_SET);

        // writes the output array into the WAV file data
        fwrite(data, wavHeader.Subchunk2Size, 1,wavFile);

        // closes the output WAV file
        fclose(wavFile);

        // stops a write clock
        end_time = clock::now();

        // casts write_run_time in nanoseconds
        auto write_run_time = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

        cout << "Time to write in nanoseconds: " << write_run_time << endl << endl;

        // free allocated memory
        delete[] inputL;
        delete[] inputR;
        delete[] outputL;
        delete[] outputR;
        delete[] data;
        inputL = nullptr;
        inputR = nullptr;
        outputL = nullptr;
        outputR = nullptr;
        data = nullptr;
    }

    // stops the program clock
    auto program_end_time = clock::now();

    // casts program_run_time in nanoseconds
    auto program_run_time = chrono::duration_cast<chrono::nanoseconds>(program_end_time - program_start_time).count();

    cout << "Total program runtime in nanoseconds: " << program_run_time << endl << endl << endl;

    return 0;
}