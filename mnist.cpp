//mnist.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist_Label(string filename,string save)
{
	ofstream saveLabel;
	saveLabel.open(save);
	fstream file(filename);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		for (int i = 0; i < number_of_images; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			saveLabel<<(int)label<<" "; //输出标签文件
		}
	}
    else{
        cout<<"open file failed."<<endl;
    }
	saveLabel.close();
	file.close();
}

void read_Mnist_Images(string filename,string path)
{
	fstream file(filename);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;
		Mat temp(n_rows,n_cols,CV_8UC1,Scalar::all(0));
		for (int i = 0; i < number_of_images; i++)
		{
			string tm=path;
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
					temp.at<uchar>(r,c)=image;
				}
			}
			imwrite(tm.append(to_string(i)).append(".jpg"),temp); //保存图片
		}
	}
    else{
        cout<<"open file failed."<<endl;
    }
	file.close();
}
 
int main()
{
	read_Mnist_Label("./mnist/t10k-labels.idx1-ubyte","./testData/testLabel.txt");
	read_Mnist_Images("./mnist/t10k-images.idx3-ubyte","./testData/");
	
	read_Mnist_Label("./mnist/train-labels.idx1-ubyte","./trainData/trainLabel.txt");
	read_Mnist_Images("./mnist/train-images.idx3-ubyte","./trainData/");

    cout<<"end."<<endl;
	return 0;
}
