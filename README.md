# Neural ODEs in Tumor Growth Modeling with Delay
### Ecuaciones Diferenciales Neuronales en un Modelo de Crecimiento Tumoral con Retardo

> Trabajo de Grado — Fundación Universitaria Konrad Lorenz  
> Programa de Matemáticas · Facultad de Matemáticas e Ingeniería · Bogotá, Mayo 2024

**Autora:** Nathalia Valentina Castiblanco Carretero  
**Director:** John Alexander Arredondo García  
**Codirector:** Miguel González Duque  

---

## 📋 Descripción

Este repositorio contiene los códigos, simulaciones y resultados del trabajo de grado que investiga la **eficacia de las Ecuaciones Diferenciales Neuronales (NDEs)** aplicadas a un modelo de crecimiento tumoral con retardo temporal.

Se estudia un modelo de alta complejidad matemática, altamente no lineal, considerado con y sin retardo, comparando los resultados obtenidos mediante la teoría clásica de sistemas dinámicos con los obtenidos a través de Neural ODEs adaptadas al modelo.

---

## 🎯 Objetivos

- Estudiar y documentar la teoría básica de las Ecuaciones Diferenciales Neuronales (NDEs).
- Aplicar la teoría de NDEs a un modelo que describe el crecimiento de un tumor con y sin retardo temporal.
- Evaluar la eficacia de las NDEs comparándolas con Redes Neuronales Recurrentes (RNNs).

---

## 🧬 El Modelo Biológico

Se trabaja con el modelo propuesto por **Jianquan Li et al. (2021)** que describe la interacción entre células tumorales (T) y células efectoras del sistema inmune (E).

**Modelo sin retardo:**

$$\frac{dT}{dt} = rT\left(1-\frac{T}{K}\right) - nET, \qquad \frac{dE}{dt} = \sigma + \mu TE - \eta E$$

**Modelo con retardo temporal τ:**

$$\frac{dT}{dt} = rT(t)\left(1-\frac{T(t)}{K}\right) - nE(t)T(t), \qquad \frac{dE}{dt} = \sigma + \mu T(t-\tau)E(t) - \eta E(t)$$

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| r | 2.5 | Tasa de crecimiento tumoral |
| K | 2.0 | Capacidad de carga |
| n | 0.8 | Tasa de destrucción tumoral por células inmunes |
| σ | 0.5 | Tasa de producción de células efectoras |
| μ | 4.0 | Tasa de estimulación inmune por tumor |
| η | 1.5 | Tasa de muerte de células efectoras |

---

## 📁 Estructura del Repositorio

```
neural-ode-tumor-growth/
│
├── 📂 notebooks/
│   ├── 📂 00_basics/
│   │   └── jax_basics.ipynb     # Introducción a JAX y Equinox
│   ├── 📂 01_figures/
│   │   ├── Figuras.ipynb                      # Figuras del modelo clásico
│   │   └── Figuras_tumor.ipynb                # Diagrama de bifurcación
│   ├── 📂 02_neural_ode/
│   │   ├── NeuralODEejemplo.ipynb             # Ejemplo introductorio
│   │   ├── NeuralODE.ipynb                    # Neural ODE sin retardo
│   │   └── NeuralODEconDelay.ipynb            # Neural ODE con retardo
│   ├── 📂 03_rnn/
│   │   └── RedNeu.ipynb                       # RNN/GRU (línea base)
│   ├── 📂 04_comparison/
│   │   ├── NDE_vs_RNN.ipynb                   # Comparación sin retardo
│   │   └── NDE_vs_RNN_delay.ipynb             # Comparación con retardo
│   └── 📂 05_ndde/
│       └── NDDE.ipynb                         # Neural Delay Differential Equations
│
├── 📂 figures/                                # Gráficas generadas
├── 📂 data/                                   # Datos de entrenamiento
├── 📂 docs/
│   ├── thesis.pdf                             # Trabajo de grado completo
│   └── slides.pdf                             # Diapositivas de sustentación
├── requirements.txt
└── README.md
```

---

## 📓 Descripción de los Notebooks

### 🔵 00 · Fundamentos
| Notebook | Descripción |
|----------|-------------|
| `jax_basics_for_Nathalia.ipynb` | Introducción práctica a JAX y Equinox: autoencoder, JIT compilation y primeros pasos con redes neuronales. Material de preparación para los modelos principales. |

### 🟢 01 · Figuras del Modelo Clásico
| Notebook | Descripción |
|----------|-------------|
| `Figuras.ipynb` | Figuras del análisis dinámico: modelo SIR, modelo tumoral sin retardo (`scipy`) y con retardo (`jitcdde`). |
| `Figuras_tumor.ipynb` | Diagrama de bifurcación del modelo tumoral en función de τ. Análisis de los 4 equilibrios de la tabla 2.1 de la tesis. |

### 🟡 02 · Neural ODEs
| Notebook | Descripción |
|----------|-------------|
| `NeuralODEejemplo.ipynb` | Ejemplo introductorio con un sistema dinámico simple. Arquitectura base con `Func` + `NeuralODE` usando Equinox y Diffrax. |
| `NeuralODE.ipynb` | Neural ODE entrenada sobre el modelo tumoral **sin retardo**. Solucionador Tsit5, entrenamiento con método adjunto, evaluación por MSE. |
| `NeuralODEconDelay.ipynb` | Neural ODE entrenada con datos del modelo tumoral **con retardo** (generados con `jitcdde`). |

### 🟠 03 · Red Neuronal Recurrente
| Notebook | Descripción |
|----------|-------------|
| `RedNeu.ipynb` | RNN con celda GRU (Equinox) sobre el modelo tumoral. Línea base de comparación contra las Neural ODEs. |

### 🔴 04 · Comparación NDE vs RNN
| Notebook | Descripción |
|----------|-------------|
| `NDE_vs_RNN.ipynb` | Comparación Neural ODE vs RNN en el modelo **sin retardo**. Las NDEs superan a las RNNs en MSE y generalización. |
| `NDE_vs_RNN_delay.ipynb` | Comparación Neural ODE vs RNN en el modelo **con retardo**. Las NDEs siguen siendo superiores aunque con mayor error. |

### 🟣 05 · Neural DDEs
| Notebook | Descripción |
|----------|-------------|
| `NDDE.ipynb` | Exploración de Neural Delay Differential Equations (NDDEs) con versión experimental de Diffrax. Trabajo hacia una solución más precisa para sistemas con memoria. |

---

## 📊 Resultados Principales

| Configuración | Modelo | Resultado |
|--------------|--------|-----------|
| Sin retardo | Neural ODE | ✅ MSE menor, mejor generalización |
| Sin retardo | RNN (GRU) | Mayor error |
| Con retardo | Neural ODE | ✅ Mejor que RNN |
| Con retardo | RNN (GRU) | Mayor error |

**Conclusión:** Las Neural ODEs superan consistentemente a las RNNs. Sin embargo, presentan limitaciones al modelar sistemas con retardo temporal, lo que motiva el desarrollo de las NDDEs.

---
## 🍺 Divulgación Científica
Este trabajo también fue presentado en formato de charla divulgativa 
bajo el título "Del enfriamiento de la cerveza a la inteligencia artificial",
explicando las ecuaciones diferenciales desde la Ley de Enfriamiento de Newton
hasta las Neural ODEs, dirigido a público general.

- 📊 [Presentación: Math & Beer](outreach/math_and_beer_talk.pptx)
- 💻 [Simulación interactiva](outreach/simulation_beer_cooling.ipynb)
  
---

## 🛠️ Instalación

```bash
git clone https://github.com/tu-usuario/neural-ode-tumor-growth.git
cd neural-ode-tumor-growth
pip install -r requirements.txt
```

**`requirements.txt`**
```
jax
jaxlib
diffrax==0.5.0
equinox==0.11.4
optax
jaxtyping
numpy
scipy
matplotlib
pandas
jitcdde
symengine
```

> ⚠️ El notebook `NDDE.ipynb` requiere una versión experimental de diffrax instalada directamente desde GitHub. Ver instrucciones dentro del notebook.

---

## 📚 Referencias

- Li, J. et al. (2021). *A tumor-immune interaction model with the effect of impulse therapy.*
- Chen, R. T. Q. et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS.
- Kidger, P. (2022). *On Neural Differential Equations.* PhD Thesis, University of Oxford.

---

## 📄 Documentos

- 📖 [Trabajo de grado completo](docs/thesis.pdf)
- 📊 [Diapositivas de sustentación](docs/slides.pdf)

---

*Fundación Universitaria Konrad Lorenz · Programa de Matemáticas · Bogotá, 2024*
