#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define THRESHOLD 128
#define WHITE 200
#define BLACK 0

void showMessage(const char *message) {
    MessageBox(NULL, message, "Info", MB_OK | MB_ICONINFORMATION);
}

int main() {
    FILE *fIn = fopen("lena512.BMP", "rb");
    FILE *fOut = fopen("b_w.BMP", "wb");

    if (fIn == NULL) {
        showMessage("Input file does not exist.");
        return 1;
    }

    unsigned char header[54];
    fread(header, sizeof(unsigned char), 54, fIn);
    fwrite(header, sizeof(unsigned char), 54, fOut);

    int width = *(int*)&header[18];
    int height = *(int*)&header[22];
    int bitDepth = *(int*)&header[28];

    unsigned char colorTable[1024];
    if (bitDepth <= 8) {
        fread(colorTable, sizeof(unsigned char), 1024, fIn);
        fwrite(colorTable, sizeof(unsigned char), 1024, fOut);
    }

    int size = width * height;
    unsigned char *buffer = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char *outputBuffer = (unsigned char*)malloc(size * sizeof(unsigned char));

    fread(buffer, sizeof(unsigned char), size, fIn);

    // Apply halftoning algorithm
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;
            int oldPixel = buffer[i];
            int newPixel = (oldPixel > THRESHOLD) ? WHITE : BLACK;
            outputBuffer[i] = newPixel;
            int error = oldPixel - newPixel;
            // Distribute the error to the neighboring pixels
            if (x + 1 < width) buffer[i + 1] += error * 0.2; // right
            if (y + 1 < height) {
                if (x > 0) buffer[i + width - 1] += error * 0.6; // bottom left
                buffer[i + width] += error * 0.1; // bottom
                if (x + 1 < width) buffer[i + width + 1] += error * 0.1; // bottom right
            }
        }
    }

    fwrite(outputBuffer, sizeof(unsigned char), size, fOut);

    fclose(fIn);
    fclose(fOut);
    free(buffer);
    free(outputBuffer);

    showMessage("Halftoning image processing completed successfully.");

    return 0;
}
