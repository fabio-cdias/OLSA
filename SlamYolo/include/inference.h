#ifndef INFERENCE_H
#define INFERENCE_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

class Inference
{
public:
    Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640}, const std::string &classesTxtFile = "", const bool &runWithCuda = true);
    std::vector<Detection> runInference(const cv::Mat &input);

private:
    void loadClassesFromFile();
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string modelPath{};
    std::string classesPath{};
    bool cudaEnabled{};

    std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    // std::vector<std::string> classes{"pessoa", "bicicleta", "carro", "motocicleta", "avião", "ônibus", "trem", "caminhão", "barco", "semáforo", "hidrante", "placa de pare", "parquímetro", "banco", "pássaro", "gato", "cachorro", "cavalo", "ovelha", "vaca", "elefante", "urso", "zebra", "girafa", "mochila", "guarda-chuva", "bolsa", "gravata", "mala", "frisbee", "esquis", "snowboard", "bola de esporte", "pipa", "taco de beisebol", "luva de beisebol", "skate", "prancha de surfe", "raquete de tênis", "garrafa", "taça de vinho", "copo", "garfo", "faca", "colher", "tigela", "banana", "maçã", "sanduíche", "laranja", "brócolis", "cenoura", "cachorro-quente", "pizza", "rosquinha", "bolo", "cadeira", "sofá", "planta em vaso", "cama", "mesa de jantar", "banheiro", "tv", "notebook", "mouse", "controle remoto", "teclado", "telefone celular", "micro-ondas", "forno", "torradeira", "pia", "geladeira", "livro", "relógio", "vaso", "tesoura", "urso de pelúcia", "secador de cabelo", "escova de dentes"};

    cv::Size2f modelShape{};

    float modelConfidenceThreshold {0.25};
    float modelScoreThreshold      {0.45};
    float modelNMSThreshold        {0.50};

    bool letterBoxForSquare = true;

    cv::dnn::Net net;
};

#endif // INFERENCE_H
