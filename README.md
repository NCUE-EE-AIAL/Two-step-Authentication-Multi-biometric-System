# Two-step-Authentication-Muti-biometric-System

## Table of Contents

- [Dataset](#dataset)
- [Architecture](#architecture)
- [How to use](#how-to-use)
- [Result](#result)
- [Acknowledge](#acknowledge)

# Dataset: 
Face image: the raw data are provided by the student in EE's class, which is revealed in dataset folder.  
Voice: LibriSpeech (Link: https://www.openslr.org/12) (train-clean-360 , test-clean)

# Architecture
## System Prototype
![workflow](https://github.com/NCUE-EE-AIAL/Two-Step-Muti-Biometric-Authentication-System/blob/main/doc/Flow_diagram.png)

## Face Recognition Model 
![face model](https://github.com/NCUE-EE-AIAL/Two-Step-Muti-Biometric-Authentication-System/blob/main/doc/graph_hr.png)

## Voice Recognition Model 
```mermaid
graph TD
    subgraph CNNs
        A2[Input Layer] --> B2[ResNet Block: filter=64]
        B2 --> C2[ResNet Block: filter=128]
        C2 --> D2[ResNet Block: filter=256]
        D2 --> G2[ResNet Block: filter=512]
    
        G2 --> N2[Reshape & Mean]
        N2 --> P2[Dense 512]
        P2 --> Q2[Output Layer]
    end

    subgraph ResNet block
        A3[Input Tensor] --> B3[Conv2D Layer: kernel_size=5]
        B3 --> C3[BatchNormalization]
        C3 --> D3[Clipped ReLU]
        D3 --> E3[Identity Block * 3]
        E3 --> F3[Output Tensor]
    end

    subgraph Identity Block
        A[Input Tensor] --> B[Conv2D Layer: kernel_size=1]
        A[Input Tensor] --> J[+]
        B --> C[BatchNorm -> Clipped ReLU]
        C --> E[Conv2D Layer: kernel_size=3]
        E --> F[BatchNorm -> Clipped ReLU]
        F --> H[Conv2D Layer: kernel_size=1]
        H --> I[BatchNormalization]
        I --> J
        J --> K[Clipped ReLU]
        K --> L[Output Tensor]
    end
```

# How to use
### image_preprocessing.py
This script processes a dataset of images to detect and crop faces using the MTCNN detector. It reads image paths from a CSV file, detects faces in each image, and crops the faces to a uniform size. The cropped face images are then saved to a specified output directory, maintaining the original subdirectory structure.

### train_face.ipynb
This script fine-tunes a pre-trained VGG16 model for a custom face recognition task. It first loads the VGG16 model with pre-trained weights, freezes its convolutional layers, and adds custom fully connected layers for classification. The model is trained on a dataset of face images, utilizing data augmentation to improve performance. The training process includes a custom callback to track batch and epoch metrics. After initial training, the convolutional layers are unfrozen, and the model is fine-tuned with a lower learning rate to further improve accuracy.

### test_face.py
This script captures an image from a webcam, detects and crops the face, and uses a pre-trained deep learning model to identify the person. It initializes the webcam, captures a frame, converts it to the required format, and saves it as a test image. The face is then cropped from the image, and the cropped face is passed to a neural network model for prediction. If the model's confidence exceeds the threshold, it prints the matching label and accuracy; otherwise, it indicates no match was found.

### voice_preprocessing.py
This script processes voice files named in the format "(speaker name)-(...)-(number)". It's configured for the Librispeech dataset and includes functions to convert .flac files to .wav. Additionally, it employs VAD (Voice Activity Detection) and Fbank (Filterbank features) to enhance model performance.

### train_voice.py
To optimize performance, training is divided into two stages: initial training with random batches to achieve acceptable results, followed by training with selected batches to refine and improve the model further. Evaluating the validation data in every epoch and show the trend diagram in the end.

### test_voice.py
This script evaluates a speaker verification model using triplet loss. It imports necessary libraries and custom modules, normalizes scores, clips audio sequences, and generates test data. The model's performance is assessed using cosine similarity, and metrics like accuracy, equal error rate, f-measure, precision, and recall are calculated.

# Result
| System                          | Accuracy   | Precision  | Recall     |
|---------------------------------|------------|------------|------------|
| Face Recognition Model          | 95.135%    | 96.317%    | 95.153%    |
| Voice Recognition Model         | 99.1%      | 88.57%     | 84.93%     |

![face result](https://github.com/NCUE-EE-AIAL/Two-Step-Muti-Biometric-Authentication-System/blob/main/doc/training_graph.png)
Graph for epochs of face recognition model showing (a) Training and validation accuracy (b) Training and validation loss.  
<br>
![voice result](https://github.com/NCUE-EE-AIAL/Two-Step-Muti-Biometric-Authentication-System/blob/main/doc/training_graph_voice.png)
Graph for epochs of voice recognition model showing (a) Training and validation EER (b) Training and validation loss.
<br>

# Acknowledge
This research is supported by TEEP (Taiwan Experience Education Program) at National Changhua University of Education
