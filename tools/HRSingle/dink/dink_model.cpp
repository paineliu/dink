// onnx.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <string>
#include <fstream>
#include "onnxruntime_cxx_api.h"
#include "dink.h"

extern unsigned short g_labels[];

char* dink_readfile(const char* path, int* length)
{
    FILE* pfile;
    char* data;

    pfile = fopen(path, "rb");
    if (pfile == NULL)
    {
        return NULL;
    }
    fseek(pfile, 0, SEEK_END);
    *length = ftell(pfile);
    data = (char*)malloc((*length) * sizeof(char));
    rewind(pfile);
    *length = fread(data, 1, *length, pfile);
    fclose(pfile);
    return data;
}

DK_HANDLE dink_init(const char *model_file)
{

    int mode_size = 0;
    char* mode = dink_readfile(model_file, &mode_size);

    // --- init onnxruntime env
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");
    
    // set options
    Ort::SessionOptions session_option;
    session_option.SetIntraOpNumThreads(5); // extend the number to do parallel
    session_option.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    Ort::Session *hHandle =  new Ort::Session(env, mode, mode_size, session_option);

    free(mode);

    return hHandle;
}
int dink_normlize(DK_POINT* norm_points, const DK_POINT* points, int point_total)
{
    int min_x = points[0].x;
    int min_y = points[0].y;
    int max_x = points[0].x;
    int max_y = points[0].y;
    int norm_point_total = 0;
    int last_x = -1;
    int last_y = -1;
    int x;
    int y;
    for (int i = 0; i < point_total; i++)
    {
        if (min_x > points[i].x) min_x = points[i].x;
        if (min_y > points[i].y) min_y = points[i].y;
        if (max_x < points[i].x) max_x = points[i].x;
        if (max_y < points[i].y) max_y = points[i].y;
    }

    int width = max_x - min_x > max_y - min_y ? max_x - min_x : max_y - min_y;

    for (int i = 0; i < point_total; i++)
    {
        x = (points[i].x - min_x) * 64 / width;
        y = (points[i].y - min_y) * 64 / width;
        if (last_x != x && last_y != y)
        {
            norm_points[norm_point_total].x = x;
            norm_points[norm_point_total].y = y;
            norm_points[norm_point_total].s = points[i].s;
            norm_points[norm_point_total].t = norm_point_total + 1;
            last_x = x;
            last_y = y;
            norm_point_total++;
        }
    }

    return norm_point_total;
}

#define TOP_K			5

typedef struct DK_VALUE_POS
{
    int nPos;
    float nValue;

} DK_VALUE_POS;

void dink_get_top_k(DK_VALUE_POS* pos, int top_k, const float* arr, int arr_size)
{
    int k = top_k;
    int j = 0;
    
    memset(pos, 0, sizeof(DK_VALUE_POS) * top_k);
    for (int i = 0; i < top_k; i++)
    {
        pos[i].nPos = -1;
        pos[i].nValue = -10000.0;
    }

    for (int i = 0; i < arr_size; i++) 
    {
        if (j == 0) 
        {
            if (arr[i] > pos[j].nValue)
            {
                pos[j].nValue = arr[i];
                pos[j].nPos = i;
                j++;
            }
        }
        else 
        {
            for (int m = 0; m < k; m++) {
                if (arr[i] > pos[m].nValue) {//顺序比较最大值
                    for (int n = 0; n < k - m - 1; n++) {//挪到位置
                        pos[k - n - 1].nValue = pos[k - n - 2].nValue;
                        pos[k - n - 1].nPos = pos[k - n - 2].nPos;
                    }
                    pos[m].nValue = arr[i];//插入到空位置
                    pos[m].nPos = i;
                    break;
                }
            }
        }
    }
}

int dink_recog(DK_HANDLE hHandle, const DK_POINT* points, int point_total, unsigned short *cands, int cand_total)
{
    DK_POINT* norm_points = new DK_POINT[point_total];
    int norm_point_total = dink_normlize(norm_points, points, point_total);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Session* session = (Ort::Session*)hHandle;
    size_t num_input_nodes = session->GetInputCount();

    // --- prepare data
    std::vector<const char*> input_names = { "hans", "hans_len" }; // must keep the same as model export
    const char* output_names[] = { "labels" };

    // use statc array to preallocate data buffer
    int han_len = norm_point_total;
    std::vector<float> input_hans_matrix;
    input_hans_matrix.resize(1 * han_len * 4);
    std::array<int64_t, 1 * 1> input_len_matrix;
    std::array<float, 1 * 3755> output_matrix;

    // must use int64_t type to match args
    std::array<int64_t, 3> input_shape{ 1, han_len, 4 };
    std::array<int64_t, 1> input_len_shape{ 1 };
    std::array<int64_t, 2> output_shape{ 1, 3755 };

    std::vector<std::vector<std::vector<float>>> sample_x;

    std::vector<std::vector<float>> pts;
    for (int i = 0; i < han_len; i++)
    {
        std::vector<float> pt;
        pt.push_back(norm_points[i].x);
        pt.push_back(norm_points[i].y);
        pt.push_back(norm_points[i].s);
        pt.push_back(norm_points[i].t);
        pts.push_back(pt);
    }

    sample_x.push_back(pts);

    // expand input as one dimention array
    for (int i = 0; i < 1; i++)
        for (int j = 0; j < han_len; j++)
            for (int k = 0; k < 4; k++)
                input_hans_matrix[i * 1 * han_len + j * 4 + k] = sample_x[i][j][k];

    std::vector<int64_t> sample_x_len = { han_len };
    input_len_matrix[0] = han_len;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_hans_matrix.data(), input_hans_matrix.size(), input_shape.data(), input_shape.size());
    Ort::Value input_len_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_len_matrix.data(), input_len_matrix.size(), input_len_shape.data(), input_len_shape.size());
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_matrix.data(), output_matrix.size(), output_shape.data(), output_shape.size());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    ort_inputs.push_back(std::move(input_len_tensor));
    // --- predict
    //Ort::Session session(env, model_path, session_option); // FIXME: must check if model file exist or valid, otherwise this will cause crash
    session->Run(Ort::RunOptions{ nullptr }, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, &output_tensor, 1); // here only use one input output channel

    // --- result
    // just use output_matrix as output, all you can use bellow code to get
    //float* output_buffer = output_tensor.GetTensorMutableData<float>();

    //std::cout << "--- predict result ---" << std::endl;
    // matrix output
    //std::cout << "ouput matrix: ";
    //for (int i = 0; i < 10; i++)
    //    std::cout << output_matrix[i] << " ";
    //std::cout << std::endl;
    // argmax value
    
    int argmax_value = std::distance(output_matrix.begin(), std::max_element(output_matrix.begin(), output_matrix.end()));
    //std::cout << "output argmax value: " << g_labels[argmax_value] << std::endl;
    DK_VALUE_POS *topk_pos = new DK_VALUE_POS[cand_total];
    dink_get_top_k(topk_pos, cand_total, output_matrix.data(), output_matrix.size());
    for (int i = 0; i < cand_total; i++)
    {
        cands[i] = g_labels[topk_pos[i].nPos];
    }
    delete[] topk_pos;
    delete []norm_points;
    return cand_total;
}

void dink_term(DK_HANDLE hHandle)
{
    delete hHandle;
}
