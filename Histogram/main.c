#include <stdio.h>
#include <stdlib.h>

#define THRESHOLD 128
#define WHITE 255
#define BLACK 0

int main() {
    FILE *fIn = fopen("lena512.bmp", "rb"); // Input File
    FILE *fOut = fopen("b_w.bmp", "wb");    // Output File

    if (fIn == NULL) { // Check if the input file has been opened successfully
        printf("Input file does not exist.\n");
        return 1;
    }

    if (fOut == NULL) { // Check if the output file can be opened
        printf("Unable to open output file.\n");
        return 1;
    }

    unsigned char header[54]; // To store the BMP header
    fread(header, sizeof(unsigned char), 54, fIn); // Read the 54-byte header
    fwrite(header, sizeof(unsigned char), 54, fOut); // Write the header to output file

    // Extract image width, height, and bit depth from Bitmap Information Header
    int width = *(int*)&header[18];
    int height = *(int*)&header[22];
    int bitDepth = *(int*)&header[28];

    printf("Width: %d\n", width);
    printf("Height: %d\n", height);
    printf("Bit Depth: %d\n", bitDepth);

    // preserve the same color palette
    unsigned char colorTable[1024];
    if (bitDepth <= 8) {
        fread(colorTable, sizeof(unsigned char), 1024, fIn);
        fwrite(colorTable, sizeof(unsigned char), 1024, fOut);
    }

    // Calculate image size (ignoring padding for simplicity, assumes no padding)
    int size = width * height;
    unsigned char *buffer = (unsigned char*)malloc(size * sizeof(unsigned char));
    int hist[256] = {0};
    // Read the pixel data
    fread(buffer, sizeof(unsigned char), size, fIn);

    // Apply thresholding to convert to black and white
    for (int i = 0; i < size; i++) {
        hist[buffer[i]]++;
    }

    //print asteriks histogram
    for (int i=0; i<256;i++){
      printf("%d", i);
      if(i<=9)printf("   |");
      else if(i>9&&i<=99)printf("  |");
      else printf(" |");
      for(int j=0;j<hist[i];j++)printf("*");
      printf("\n");
    }

    // Close files and free memory
    fclose(fIn);

    return 0;
}
