/**
 * @Author: fengqi
 * @Email: 2607546441@qq.com
*/
#include <iostream>
#include <time.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "network.h"

using namespace std;
using namespace cv;

void saveWeight(string file,Network *network){ //保存各层权重到文件
    ofstream outfile(file);
    for(int i=0;i<network->mNumLayers;i++){
        Layer *layer=network->mLayers[i];
        for(int m=0;m<layer->mNumNodes;m++){
            for(int n=0;n<layer->mNumInputNodes+1;n++){
                outfile<<layer->mWeights[m][n]<<" ";
            }
        }
    }
    outfile.close();
    cout<<"save weight file to <"<<file<<"> done."<<endl;
}

void loadWeight(string file,Network *network){ //加载权重文件
    ifstream infile(file);
    if(!infile.is_open()){
        cout<<"open weight file failed!"<<endl;
        exit(-1);
    }
    for(int i=0;i<network->mNumLayers;i++){
        Layer *layer=network->mLayers[i];
        for(int m=0;m<layer->mNumNodes;m++){
            for(int n=0;n<layer->mNumInputNodes+1;n++){
                infile>>layer->mWeights[m][n];
            }
        }
    }
    infile.close();
    cout<<"load weight from <"<<file<<"> done."<<endl;
}

float train(Network *network,string path,int imageSize, int numImages) //训练网络，使用训练数据集
{
    srand(time(0));
    float *temp = new float[imageSize];
    string la=path;
    ifstream labelFile(path.append("trainLabel.txt")); //标签文件
    int label;

    for (int i = 0; i < numImages; i++)
    {
        if(i%(numImages/10)==0){ //每 6000 张图片统计错误率，并显示训练进度
            network->mErrorSum=0;
            cout << setfill('=') << setw(2) << ">"<<(i/(numImages/10))*10<<"%"<<flush;
        }
        if(i==numImages-1)
            cout<<"====>100%"<< endl;
        int k=rand()%numImages; //随机选取图片训练
        string l=la;
        Mat x=imread(l.append(to_string(k)).append(".jpg"),0); //使用 opencv 读取图片
        if(!x.data){cout<<"read image error."<<endl;return -1;}
        for(int m=0;m<x.rows;m++){
            for(int n=0;n<x.cols;n++){
                float a=(x.at<uchar>(m,n))/255.0; //归一化
                temp[m*x.cols+n]=a;
            }
        }
        labelFile.seekg(2*k); //标签和图片对应
        labelFile>>label;
        network->compute(temp,label); //每次训练一张图片
    }
    cout << "the error is:" << network->mErrorSum/(numImages/10);
    labelFile.close();
    delete [] temp;
    return network->mErrorSum;
}

int validate(Network *network,string path,int imageSize, int numImages) //验证网络准确率，使用测试数据集
{
    int ok_cnt = 0;
    float* temp = new float[imageSize];
    string la=path;
    ifstream labelFile(path.append("testLabel.txt")); //标签文件
    int label,idx=0;

    for (int i = 0; i < numImages; i++)
    {
        if(i%(numImages/10)==0) //显示进度
            cout << setfill('=') << setw(2) << ">"<<(i/(numImages/10))*10<<"%"<<flush;
        if(i==numImages-1)
            cout<<"====>100%"<< endl;
        string l=la;
        Mat x=imread(l.append(to_string(i)).append(".jpg"),0); //按顺序读取图片测试
        if(!x.data){cout<<"read image error."<<endl;return -1;}
        for(int m=0;m<x.rows;m++){
            for(int n=0;n<x.cols;n++){
                float a=(x.at<uchar>(m,n))/255.0; //归一化
                temp[m*x.cols+n]=a;
            }
        }
        labelFile>>label;
        network->compute(temp,label); //验证
        float *out=network->mOutputs; //获得计算输出
        float max_value = -9999;
        for (int j = 0; j < network->mNumOutputs; j++)
        {
            if (out[j] > max_value)
            {
                max_value = out[j]; //最大输出位置即图片所属类别
                idx = j;
            }
        }
        if (idx == label) //判断与标签是否相同，即预测是否准确
        {
            ok_cnt++;
        }
    }
    labelFile.close();
    delete [] temp;
    return ok_cnt;
}

int main(int argc, char* argv[]) //主入口函数
{
    if(argc<2||(strcmp(argv[1],"train")!=0&&strcmp(argv[1],"test")!=0)){ //判断调用参数是否合法
        cout<<"usage: ./run [train/test] [weight_file]\nwrong parameter!!!"<<endl;
        return -1;
    }

    bool load_weight=false; //是否加载权重文件
    int imageRow=28,imageCol=28; //输入图片大小
    int imageSize=imageRow*imageCol;
    int trainNumImages = 60000; //训练集大小
    int testNumImages = 10000; //测试集大小

    int networkInputs=imageSize; //网络参数设置
    int networkOutputs=10;
    int epoches=10;
    float learningRate=0.1;

    Network *network = new Network(epoches,learningRate,networkInputs,networkOutputs);
    network->addLayer(256,SIGMOID); //加入全连接层，参数有神经元个数和激活函数类型
    network->addLayer(128,SIGMOID);
    network->addLayer(network->mNumOutputs,SIGMOID);
    
    cout <<"\nnetwork framework: (input)"<< network->mNumInputs;
    for(int i=0;i<network->mNumLayers;i++){
        cout<<"=>"<<network->mLayers[i]->mNumNodes;
    }
    cout<<"(output)"<<endl<<endl;

    if(argc>2){ //加载预训练权重文件
        loadWeight(argv[2],network);
        load_weight=true;
    }

    if(strcmp(argv[1],"train")==0) //训练网络
    {
        time_t time0=time(0);
        cout<<"start training..."<<endl<<endl;
        cout<<"total epoches: "<<network->mEpoches<<", NO.1 epoches. begin learning rate: "<<network->mLearningRate<< endl;
        for(int i = 0; i < network->mEpoches; i++) //共训练 epoches 轮次
        {
            string weightFile="backup/mnist.weight_";
            time_t time1=time(0);
            network->mTrain=true; //训练标志
            cout<<"\nep: "<<i+1<<", lr: "<<network->mLearningRate<<" ";
            float err = train(network,"trainData/",imageSize,trainNumImages); //开始训练
            cout<<", cost time: "<<time(0)-time1<<" seconds."<<endl;

            network->mTrain=false; //验证测试标志
            cout<<"\nvalidate...";
            int ok = validate(network,"testData/",imageSize,testNumImages); //开始验证
            cout<<"validate accuracy: "<<(float)ok/testNumImages*100<< "%, true: "<<ok<<", total: "<<testNumImages<< endl;
            if(network->mLearningRate>0.01) network->mLearningRate-= 0.01; //学习率变化调整
            else network->mLearningRate=0.01;
            if(i<network->mEpoches-1)
                saveWeight(weightFile.append(to_string(i+1)).append("_").append(to_string(ok)),network); //一轮训练结束，保存权重文件
        }
        saveWeight("mnist.weight",network); //网络训练结束，保存权重文件
        cout<<"\ntraining network success...cost time: "<<(time(0)-time0)<<" seconds.\n"<<endl;
    }
    else if(strcmp(argv[1],"test")==0) //测试预测图片
    {
        if(!load_weight){ //必须先加载网络权重
            cout<<"no weight file loaded in, can't start prediction.\n"<<endl;
            return -1;
        }
        string name;
        while(1){ //循环测试图片
            cout<<"\nplease enter the image path...(ctrl-c to exit.)"<<endl;
            getline(cin,name); //输入图片名
            Mat m=imread(name,0); //使用 opencv 读入图片
            if(!m.data){
                cout<<"read image wrong. please check image file name..."<<endl;
                continue;
            }
            network->mTrain=false;
            if(m.cols!=imageCol||m.rows!=imageRow) resize(m,m,Size(imageCol,imageRow)); //resize 图片到网络接受输入大小
            float *d=new float[imageSize];
            for(int i=0;i<imageRow;i++){
                for(int j=0;j<imageCol;j++){
                    float x=(m.at<uchar>(i,j))/255.0; //将二维像素值转成一维向量，并归一化
                    d[i*imageCol+j]=x;
                }
            }
            float max=-9999;
            int idx=10;
            network->compute(d); //开始预测
            float *out=network->mOutputs; //获得网络输出
            for(int i=0;i<network->mNumOutputs;i++){
                if(out[i]>max){ //取最大输出为预测值
                    max=out[i];
                    idx=i;
                }
            }
            cout<<"the prediction is: "<<idx<<endl;
            delete [] d;
        }
    }

    delete network;
    return 0;
}
