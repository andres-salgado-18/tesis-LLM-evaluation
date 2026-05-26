# tesis-LLM-evaluation
# Guía de Operación: LLM Experimentation Pipeline

Código para la generación de respuestas, embeddings y análisis. Para la parte de generación se usa la API de OpenAI, mientras que para la generación embeddings (tanto del prompt como de la respuesta) se pueden generar de forma local (`sentence-transformers`) y mediante la API de OpenAI.

Para utilizar los modelos vía OpenAI API, es necesario definir la variable de entorno `OPENAI_API_KEY`. El código maneja esto mediante la lectura de un archivo `.env`.

## Comandos de Ejecución para Generación

### 1. Generación de Texto (OpenAI Batch API)

Este modo envía prompts para procesamiento asíncrono (batches).
#### Ejecución estándar:

``` bash
python main.py --task generate --input data/dataset.csv --gen_model gpt-4o-mini --temp 0.1
```

#### Modo de Recuperación:

Si el script se interrumpe, se puede retomar el proceso usando el ID del batch de la API de OpenAI:

``` bash
python main.py --task generate --input data/dataset.csv --batch_id batch_abc123
```

------------------------------------------------------------------------

### 2. Generación de Embeddings

Transforma columnas de texto en vectores numéricos. El sistema detecta
automáticamente si el modelo es local o API.

#### Modelos Locales (Sentence-Transformers):

``` bash
python main.py --task embed --input outputs/generaciones.parquet --emb_model all-MiniLM-L6-v2 --batch_size 128
```

#### Modelos vía API (OpenAI):

``` bash
python main.py --task embed --input outputs/generaciones.parquet --emb_model text-embedding-3-small --batch_size 64
```

------------------------------------------------------------------------

### Parámetros de la CLI

| Argumento       | Descripción                                             | Por defecto              |
|----------------|---------------------------------------------------------|--------------------------|
| `--task`       | Tarea a ejecutar: `generate` o `embed`                  | Requerido                |
| `--input`      | Ruta al archivo CSV o Parquet de entrada                | Requerido                |
| `--batch_id`   | ID de Batch existente para recuperar resultados         | `None`                   |
| `--gen_model`  | Modelo de OpenAI para generación                        | `gpt-4o-mini`            |
| `--temp`       | Temperatura (0.0 a 2.0)                                 | `0.1`                    |
| `--max_tokens` | Límite de tokens por respuesta                          | `500`                    |
| `--emb_model`  | Modelo para embeddings (local o API)                    | `text-embedding-3-small` |
| `--batch_size` | Tamaño del lote de procesamiento                        | `64`                     |



## Análisis (en desarrollo)
La parte de análisis por ahora se está llevando a cabo en el jupyter notebook `Analysis`. Cuando estén definidas las métricas y gráficos a utilizar, el código pasará mayormente a `src/analysis` y `src/metrics`.

