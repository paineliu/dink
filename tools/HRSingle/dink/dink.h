#ifndef __DINK_H__
#define __DINK_H__

typedef struct _DK_POINT
{
    int x;
    int y;
    int s;
    int t;
} DK_POINT;

typedef void* DK_HANDLE;

#define DK_FALSE 0
#define DK_TRUE 0

DK_HANDLE dink_init(const char* model_file);

int dink_recog(DK_HANDLE hHandle, const DK_POINT* points, int point_total, unsigned short* cands, int cand_total);

void dink_term(DK_HANDLE hHandle);

#endif