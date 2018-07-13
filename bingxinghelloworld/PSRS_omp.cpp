#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<algorithm>
#include<fstream>
#include<iostream>

using namespace std;

const int MBSIZE = 1024 * 1024;

int **temp;

int **segment;    //各处理器按照住院的各自的有序段号
int *sizes;
int *sample;
int *pivot_number;


void PSRS(int *data, int size, int numThread);

int main() {
	int num = 512;
	int *data = (int*)malloc(sizeof(int)*(num*MBSIZE));
	ifstream fin("../data512M.txt", ios::binary);
	fin.read((char*)data, num*MBSIZE * sizeof(int));
	fin.close();
	
	int numThread = 1;
	//初始化动态数组

	// 进行排序
	double startTime = omp_get_wtime();
	PSRS(data, num*MBSIZE, numThread);
	double endTime = omp_get_wtime();

	// 检查排序
	for (int i = 1; i<num*MBSIZE; i++) {
		if (data[i]<data[i - 1]) {
			cout << "false";
			cout << data[i] << " " << data[i - 1]<<endl;
			break;
		}
	}
	cout << "数据集:" << num << "M" << ",线程数:" << numThread << endl;
	cout << "运行时间：" << endTime - startTime << " s"<<endl;

	// 释放动态数据

	free(data);

	for (int i = 0; i<numThread; i++) {
		free(temp[i]);
		free(segment[i]);
	}

	return 0;
}


void PSRS(int *data, int size, int numThread) {
	int localN = size / numThread;
	sample = (int*)malloc(sizeof(int)*(numThread*numThread));
	pivot_number = (int*)malloc(sizeof(int)*(numThread - 1));

	// 均匀划分+局部排序+ 正则采样
#pragma omp parallel num_threads(numThread)
	{
		int myRank = omp_get_thread_num();
		int localLeft = myRank*localN;
		int localRight = (myRank + 1)*localN;
		int step = localN / numThread;
		sort(data + localLeft, data + localRight);
		for (int i = 0; i < numThread; i++) {
			sample[myRank*numThread + i] = *(data + (myRank*localN + i*step));
		}
	}
	//样本排序
	sort(sample, sample + numThread*numThread);

	//选择主元
	for (int i = 1; i < numThread; i++) {
		pivot_number[i - 1] = sample[i*numThread];
	}
	segment = (int**)malloc(sizeof(int*)*numThread);
	for (int i = 0; i < numThread; i++) {
		segment[i] = (int*)malloc(sizeof(int)*(numThread + 1));
	}
	//主元划分
#pragma omp parallel num_threads(numThread)
	{
		int myRank = omp_get_thread_num();
		int localLeft = myRank*localN;
		int localRight = (myRank + 1)*localN;
		int count = 0;
		int mleft = localLeft;
		segment[myRank][count] = 0;
		segment[myRank][numThread] = localN;
		for (; mleft < localRight && count < numThread - 1;) {
			if (*(data + mleft) <= pivot_number[count]) {
				mleft += 1;
			}
			else {
				count += 1;
				segment[myRank][count] = mleft - localLeft;
			}
		}
		for (; count < numThread - 1;count++) {
			
			segment[myRank][count+1] = mleft - localLeft;
		}
	}
	// 释放动态数据
	free(sample);
	free(pivot_number);
	//sizes = (int*)malloc(sizeof(int)*numThread);
	sizes = (int*)malloc(sizeof(int)*numThread);
	temp = (int**)malloc(sizeof(int*)*numThread);
	//全局交换
	// 计算每一段的大小，动态初始化
	for (int i = 0; i < numThread; i++) {
		sizes[i] = 0;
		for (int j = 0; j < numThread; j++) {
			sizes[i] += (segment[j][i + 1] - segment[j][i]);
			//cout << sizes[i] << endl;
		}
		temp[i] = (int*)malloc(sizeof(int)*sizes[i]);
		int index = 0;
		for (int j = 0; j < numThread; j++) {
			for (int k = segment[j][i]; k < segment[j][i + 1]; k++) {
				data[localN*j + k];
			
				temp[i][index] = data[localN*j + k];
				index += 1;
			}
		}
	}
	//归并排序
#pragma omp parallel num_threads(numThread)
	{
		int myRank = omp_get_thread_num();
		// 可进行修改进一步加速，省去重新赋值的成本
		sort(temp[myRank], temp[myRank] + sizes[myRank]);
	}
	int i = 0;
	for (int j = 0; j < numThread; j++) {
		for (int k = 0; k < sizes[j]; k++) {
			*(data + i) = *(temp[j] + k);
			i++;
		}
	}
	free(sizes);

}/**/
/**/