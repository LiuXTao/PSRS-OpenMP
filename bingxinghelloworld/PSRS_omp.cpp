#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<algorithm>
#include<fstream>
#include<iostream>

using namespace std;

const int MBSIZE = 1024 * 1024;

int **temp;

int **segment;    //������������סԺ�ĸ��Ե�����κ�
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
	//��ʼ����̬����

	// ��������
	double startTime = omp_get_wtime();
	PSRS(data, num*MBSIZE, numThread);
	double endTime = omp_get_wtime();

	// �������
	for (int i = 1; i<num*MBSIZE; i++) {
		if (data[i]<data[i - 1]) {
			cout << "false";
			cout << data[i] << " " << data[i - 1]<<endl;
			break;
		}
	}
	cout << "���ݼ�:" << num << "M" << ",�߳���:" << numThread << endl;
	cout << "����ʱ�䣺" << endTime - startTime << " s"<<endl;

	// �ͷŶ�̬����

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

	// ���Ȼ���+�ֲ�����+ �������
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
	//��������
	sort(sample, sample + numThread*numThread);

	//ѡ����Ԫ
	for (int i = 1; i < numThread; i++) {
		pivot_number[i - 1] = sample[i*numThread];
	}
	segment = (int**)malloc(sizeof(int*)*numThread);
	for (int i = 0; i < numThread; i++) {
		segment[i] = (int*)malloc(sizeof(int)*(numThread + 1));
	}
	//��Ԫ����
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
	// �ͷŶ�̬����
	free(sample);
	free(pivot_number);
	//sizes = (int*)malloc(sizeof(int)*numThread);
	sizes = (int*)malloc(sizeof(int)*numThread);
	temp = (int**)malloc(sizeof(int*)*numThread);
	//ȫ�ֽ���
	// ����ÿһ�εĴ�С����̬��ʼ��
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
	//�鲢����
#pragma omp parallel num_threads(numThread)
	{
		int myRank = omp_get_thread_num();
		// �ɽ����޸Ľ�һ�����٣�ʡȥ���¸�ֵ�ĳɱ�
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