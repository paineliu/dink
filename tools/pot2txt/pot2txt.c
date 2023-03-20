#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#define SAMPLE_DATA_MAX_SIZE (1024 * 64)
typedef struct _STROKE_POINT
{
	short x;
	short y;

} STROKE_POINT;

int pot2txt(const char* json_filename, const char* txt_filename)
{
	unsigned char sample_data[SAMPLE_DATA_MAX_SIZE];
	FILE* fp = fopen(json_filename, "rb");
	FILE* fp_txt = fopen(txt_filename, "w");
	//printf("%s -> %s\r", json_filename, txt_filename);

	if (fp && fp_txt)
	{
		unsigned short sample_size;
		while (fread(&sample_size, sizeof(sample_size), 1, fp))
		{
			if (sample_size < SAMPLE_DATA_MAX_SIZE)
			{
				fread(sample_data, sizeof(unsigned char), sample_size - 2, fp);
				unsigned int* code = (unsigned int*)&sample_data[0];
				unsigned short* stroke_num = (unsigned short*)&sample_data[4];
				STROKE_POINT* points = (STROKE_POINT*)&sample_data[6];
				int point_total = (sample_size - 8) / 4;

				char code_str[3];
				code_str[0] = sample_data[1];
				code_str[1] = sample_data[0];
				code_str[2] = 0;
				fprintf(fp_txt, "%s\t", code_str);
				
				for (int i = 0; i < point_total; i++)
				{
					if (i == point_total - 1)
					{
						fprintf(fp_txt, "%d,%d\n", points[i].x, points[i].y);
					}
					else
					{
						fprintf(fp_txt, "%d,%d,", points[i].x, points[i].y);
					}
				}
			}
			else
			{
				printf("error: sample_data out of size = %d.", sample_size);
				break;
			}
		}
	}
	if (fp_txt)
	{
		fclose(fp_txt);
	}

	if (fp)
	{
		fclose(fp);
	}

	return 0;
}

int main(int argc, char* argv[])
{
	if (argc == 2) {
		char txt_filename[1024];
		sprintf(txt_filename, "%s.txt", argv[1]);
		pot2txt(argv[1], txt_filename);
	}
	else if (argc == 3) {
		pot2txt(argv[1], argv[2]);
	}
	else
	{
		printf("%s pot_filename txt_filename.\n", argv[0]);
	}

	return 0;
}