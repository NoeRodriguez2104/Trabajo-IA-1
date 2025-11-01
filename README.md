# Modelos de Hugging Face usados en MusicStudio AI 🎧

Este proyecto utiliza **dos modelos principales de Hugging Face**, cada uno especializado en una tarea diferente de inteligencia artificial aplicada al **análisis musical y auditivo**.

---

## 1. Modelo ASR — `openai/whisper-large-v3`

**Tarea:** Reconocimiento Automático del Habla (ASR)  
**Entrada:** Audio (voz o canción con letra)  
**Salida:** Texto transcrito (letra detectada o narración hablada)

### Descripción

`openai/whisper-large-v3` es una versión profesional del modelo **Whisper** desarrollado por **OpenAI**.  
Está entrenado en una gran cantidad de audio multilingüe y es capaz de transcribir voz humana incluso en condiciones de ruido.  
En este proyecto se utiliza para **extraer la letra** o **texto hablado** de un fragmento musical.

### Detalles técnicos

- **Tamaño del modelo:** 500-600 MB
- **Parámetros:** 1550
- **Idiomas soportados:** más de 90 idiomas (incluye español)
- **Ventajas:** rápido, ideal para ejecución en CPU o entornos educativos
- **Instalación de FFMPEG (requisito previo) con Chocolatey:**

```
choco install ffmpeg -y

```

- **Requisitos previos:** 

- **Instalación de python 3.11.9 en la máquina del usuario** => https://www.python.org/downloads/release/python-3119/

- **Instalación de CUDA 12.1** => https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local  

```
winget install Chocolatey
choco install ffmpeg -y

```

```
#Crear entorno virtual 
py -3.11 -m venv .venv
#Instacion de dependencias
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install transformers soundfile librosa ipywidgets jupyterlab huggingface-hub accelerate python-dotenv
#Instalacion local de Demucs (modelo para separar las pistas de audio)
python3 -m pip install -U demucs

```
