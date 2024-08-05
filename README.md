# TEEP2023_Mutimodal_Biometric_Authentication
# Dataset: 
Face image: the raw data are provided by the student in EE's class, which is revealed in dataset folder.  
Voice: LibriSpeech (Link: https://www.openslr.org/12) (train-clean-360 , test-clean)

# Face Recognition
## Model Architecture
![face model](https://github.com/NCUE-EE-AIAL/Two-Step-Muti-Biometric-Authentication-System/blob/main/doc/graph_hr.png)

## How to use


# Voice Recognition
## Model Architecture
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
## How To Use
### pre_process.py
This script processes voice files named in the format "(speaker name)-(...)-(number)". It's configured for the Librispeech dataset and includes functions to convert .flac files to .wav. Additionally, it employs VAD (Voice Activity Detection) and Fbank (Filterbank features) to enhance model performance.

### train.py
To optimize performance, training is divided into two stages: initial training with random batches for 20 epochs to achieve acceptable results, followed by 40 epochs of training with selected batches to refine and improve the model further.

# Result
![face result](https://github.com/NCUE-EE-AIAL/Two-Step-Muti-Biometric-Authentication-System/blob/main/doc/training_graph.png)
Graph for epochs of face recognition model showing (a) Training and validation accuracy (b) Training and validation loss.  
  
  
![voice result](https://github.com/NCUE-EE-AIAL/Two-Step-Muti-Biometric-Authentication-System/blob/main/doc/training_graph_voice.png)
Graph for epochs of voice recognition model showing (a) Training and validation EER (b) Training and validation loss.

# Acknowledge
This research is supported by TEEP (Taiwan Experience Education Program)
