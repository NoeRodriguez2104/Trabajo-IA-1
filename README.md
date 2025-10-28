# Modelos de Hugging Face usados en MusicStudio AI 🎧

Este proyecto utiliza **dos modelos principales de Hugging Face**, cada uno especializado en una tarea diferente de inteligencia artificial aplicada al **análisis musical y auditivo**.

---

## 1. Modelo ASR — `openai/whisper-tiny`

**Tarea:** Reconocimiento Automático del Habla (ASR)  
**Entrada:** Audio (voz o canción con letra)  
**Salida:** Texto transcrito (letra detectada o narración hablada)

### Descripción
`openai/whisper-tiny` es una versión ligera del modelo **Whisper** desarrollado por **OpenAI**.  
Está entrenado en una gran cantidad de audio multilingüe y es capaz de transcribir voz humana incluso en condiciones de ruido.  
En este proyecto se utiliza para **extraer la letra** o **texto hablado** de un fragmento musical.

### Detalles técnicos
- **Tamaño del modelo:** ~39 MB  
- **Parámetros:** 39 millones  
- **Idiomas soportados:** más de 90 idiomas (incluye español)  
- **Ventajas:** rápido, ideal para ejecución en CPU o entornos educativos  
- **Pipeline utilizado:**  
  ```python
  from transformers import pipeline
  asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
  result = asr_pipeline("audio.mp3")
  print(result["text"])
