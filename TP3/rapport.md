# CSC8607 — Introduction au Deep Learning

## TP3

---

# Exercice 1 

## Question 1

Sans `transforms.Normalize`, l'entraînement reste mathématiquement possible, mais les valeurs données au réseau ne sont plus centrées et mises à une échelle adaptée.

Cela peut rendre l'optimisation moins stable et ralentir la convergence, car les activations et les gradients peuvent avoir des échelles moins favorables à la descente de gradient.

Les moyennes et écarts-types doivent être calculés uniquement sur le jeu d'entraînement. Les calculer aussi avec les données de validation introduirait une information provenant de la validation dans le prétraitement du train : ce serait une fuite de données (`data leakage`).

---

## Question 2

Pour l'entraînement, nous utilisons :

```python
RandomResizedCrop(227)
RandomHorizontalFlip()
```

Ces transformations sont aléatoires et permettent de présenter au réseau plusieurs variantes des images d'entraînement. Cela augmente la diversité des exemples et limite la mémorisation du dataset.

Pour la validation, nous utilisons :

```python
Resize(256)
CenterCrop(227)
```

sans transformation aléatoire.

La validation reste ainsi déterministe : une image est évaluée de la même manière à chaque époque, ce qui permet de comparer correctement les performances du modèle.

---

## Question 3

Un exemple est la classification de panneaux indiquant une direction :

```text
Tourner à gauche
Tourner à droite
```

Un miroir horizontal peut transformer visuellement un panneau « gauche » en panneau « droite », alors que le label reste inchangé.

Dans ce cas, l'augmentation de données créerait donc un exemple avec une mauvaise étiquette.

---

# Exercice 2

## Etape 1

### Question 1
 
L'image d'entrée possède :

```math
3 \times 227 \times 227 = 154587
```

valeurs.

L'architecture du MLP est donc :

```text
154587 -> 256 -> 256 -> 1000
```

#### Nombre de paramètres

Pour la première couche :

```math
154587 \times 256 = 39574272
```

Avec les biais :

```math
39574272 + 256 = 39574528
```

Pour la deuxième couche :

```math
256 \times 256 + 256 = 65792
```

Pour la couche de sortie :

```math
256 \times 1000 + 1000 = 257000
```

On obtient :

| Couche | Poids | Biais | Total |
| --- | ---: | ---: | ---: |
| `154587 -> 256` | 39 574 272 | 256 | 39 574 528 |
| `256 -> 256` | 65 536 | 256 | 65 792 |
| `256 -> 1000` | 256 000 | 1 000 | 257 000 |
| **Total** | **39 895 808** | **1 512** | **39 897 320** |

Le script PyTorch confirme :

```text
Paramètres avec biais : 39 897 320
Paramètres sans biais : 39 895 808
```

La première couche contient :

```math
\frac{39574528}{39897320} \times 100
\approx 99.19\%
```

des paramètres du réseau.

C'est donc cette première couche qui concentre plus de 99 % des paramètres.

### Question 2

Le problème principal est l'utilisation de `flatten`.

L'image 2D est transformée en un grand vecteur, ce qui fait perdre explicitement sa structure spatiale. Le MLP ne profite donc pas naturellement des relations locales entre pixels voisins.

Cela explique pourquoi cette architecture est moins adaptée aux images qu'un réseau convolutif.

---

## Etape 2
### Question 2.b

Les dimensions obtenues dans AlexNet sont :

```text
227 × 227
-> Conv1    : 56 × 56
-> MaxPool1 : 27 × 27
-> Conv2    : 27 × 27
-> MaxPool2 : 13 × 13
-> Conv3    : 13 × 13
-> Conv4    : 13 × 13
-> Conv5    : 13 × 13
-> MaxPool3 : 6 × 6
```

Après le dernier pooling :

```math
256 \times 6 \times 6 = 9216
```

### Tableau complété

| Couche | Taille entrée (C × H × W) | Filtres / neurones | K | S | P | Taille sortie (C × H × W) | Paramètres avec biais | Mémoire sortie (KB) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Conv1 + ReLU | `3 × 227 × 227` | 64 | 11×11 | 4 | 2 | `64 × 56 × 56` | 23 296 | 784.00 |
| MaxPool1 | `64 × 56 × 56` | - | 3×3 | 2 | 0 | `64 × 27 × 27` | 0 | 182.25 |
| Conv2 + ReLU | `64 × 27 × 27` | 192 | 5×5 | 1 | 2 | `192 × 27 × 27` | 307 392 | 546.75 |
| MaxPool2 | `192 × 27 × 27` | - | 3×3 | 2 | 0 | `192 × 13 × 13` | 0 | 126.75 |
| Conv3 + ReLU | `192 × 13 × 13` | 384 | 3×3 | 1 | 1 | `384 × 13 × 13` | 663 936 | 253.50 |
| Conv4 + ReLU | `384 × 13 × 13` | 256 | 3×3 | 1 | 1 | `256 × 13 × 13` | 884 992 | 169.00 |
| Conv5 + ReLU | `256 × 13 × 13` | 256 | 3×3 | 1 | 1 | `256 × 13 × 13` | 590 080 | 169.00 |
| MaxPool3 | `256 × 13 × 13` | - | 3×3 | 2 | 0 | `256 × 6 × 6` | 0 | 36.00 |
| Flatten + Dropout | `256 × 6 × 6` | - | - | - | - | `9216` | 0 | 36.00 |
| Linear1 + ReLU + Dropout | `9216` | 4096 | - | - | - | `4096` | 37 752 832 | 16.00 |
| Linear2 + ReLU | `4096` | 4096 | - | - | - | `4096` | 16 781 312 | 16.00 |
| Linear3 | `4096` | 1000 | - | - | - | `1000` | 4 097 000 | 3.91 |

La mémoire totale des activations de sortie est d'environ :

```math
2339.16\ \text{KB}
\approx
2340\ \text{KB par image}
```

### Répartition des paramètres et des activations

Les couches fully-connected contiennent :

```text
Linear1 : 37 752 832
Linear2 : 16 781 312
Linear3 :  4 097 000
```

soit :

```math
58631144
```

paramètres sur un total de :

```math
61100840
```

Cela représente environ :

```math
95.96\%
```

des paramètres d'AlexNet.

L'essentiel des **poids** se trouve donc dans les couches fully-connected.

À l'inverse, la mémoire des **activations** est surtout importante dans les premières couches convolutionnelles, car les cartes de caractéristiques sont encore grandes.

Par exemple :

```text
Conv1 : 784 KB
Conv2 : 546.75 KB
```

### Comparaison avec le MLP

Pour les sorties des trois couches linéaires du MLP :

```math
(256 + 256 + 1000) \times 4
=
6048\ \text{octets}
```

soit environ :

```math
5.91\ \text{KB}
```

contre environ :

```math
2340\ \text{KB}
```

pour AlexNet.

Même si les deux réseaux ont un nombre de paramètres du même ordre de grandeur, leur utilisation de la mémoire est très différente.

Le MLP concentre beaucoup de paramètres dans une grande matrice de poids, mais produit peu d'activations.

AlexNet partage les poids de ses filtres convolutifs, mais produit de grandes cartes de caractéristiques intermédiaires, qui occupent davantage de mémoire.

---

## Question 2.c

L'implémentation PyTorch donne :

```text
Paramètres AlexNet : 61 100 840
Sortie après convolutions : (1, 256, 6, 6)
Taille après flatten : 9216
Sortie finale : (1, 1000)
```

Ces valeurs sont cohérentes avec les calculs précédents.

---

# Exercice 3 

## Question 3.a

### 1. `model.train()` et `model.eval()`

AlexNet contient des couches :

```python
nn.Dropout(p=0.5)
```

`model.train()` active le Dropout pendant l'entraînement.

`model.eval()` le désactive pendant la validation.

Si on oubliait `model.eval()`, le Dropout resterait actif pendant l'évaluation. Certaines activations seraient encore supprimées aléatoirement et les métriques de validation seraient moins stables et moins fiables.

---

### 2. Valeurs théoriques à l'initialisation

ImageNet possède 1000 classes.

Pour un choix complètement aléatoire, l'accuracy théorique vaut :

```math
\frac{1}{1000}
=
0.001
=
0.1\%
```

La `CrossEntropyLoss` théorique pour une distribution uniforme vaut :

```math
-\log\left(\frac{1}{1000}\right)
=
\log(1000)
\approx
6.9078
```

Pour AlexNet, nous avons obtenu à la première époque :

```text
Train Loss = 6.9080
Train Acc  = 0.0008

Val Loss   = 6.9059
Val Acc    = 0.0010
```

Ces valeurs sont presque identiques aux valeurs théoriques attendues à l'initialisation.

Pour le MLP :

```text
Train Loss = 7.1932
Train Acc  = 0.0012

Val Loss   = 7.0085
Val Acc    = 0.0012
```

L'accuracy reste également du même ordre de grandeur que le choix aléatoire.

Les premières époques sont donc cohérentes avec les valeurs théoriques.

---

### 3. Comparaison entre MLP et AlexNet sur 20 000 images

#### Loss

![Loss MLP vs AlexNet](figures/TensorBoard_Loss_MLP_vs_AlexNet.png)

#### Accuracy

![Accuracy MLP vs AlexNet](figures/TensorBoard_Accuracy_MLP_vs_AlexNet.png)

À l'époque 10 :

| Modèle | Train Loss | Val Loss | Train Accuracy | Val Accuracy |
| --- | ---: | ---: | ---: | ---: |
| MLP | 6.7973 | 6.8939 | 0.62 % | 0.42 % |
| AlexNet | 6.4079 | 6.4746 | 0.98 % | 0.80 % |

AlexNet apprend mieux et plus rapidement que le MLP : sa loss diminue davantage et ses accuracies sont supérieures.

Cela est cohérent avec son architecture convolutive, qui exploite la structure spatiale des images.

Sur ces 10 époques, aucun des deux modèles ne montre encore un sur-apprentissage très fort.

Le MLP présente cependant un écart train/validation légèrement plus visible à la fin :

```text
Train Loss = 6.7973
Val Loss   = 6.8939
```

Il est aussi plus exposé au sur-apprentissage sur un petit dataset, car sa première couche contient à elle seule près de 40 millions de paramètres sans exploiter la structure locale des images.

---

## Question 3.b - AlexNet plus profond

Nous avons ajouté trois convolutions `3 × 3` supplémentaires après `Conv5`.

Le nombre de paramètres passe de :

```text
AlexNet standard : 61 100 840
Deep AlexNet     : 62 871 080
```

Après 10 époques sur les mêmes 20 000 images :

| Modèle | Train Loss | Val Loss | Train Accuracy | Val Accuracy |
| --- | ---: | ---: | ---: | ---: |
| AlexNet standard | 6.4079 | 6.4746 | 0.98 % | 0.80 % |
| Deep AlexNet | 6.5329 | 6.5937 | 0.56 % | 0.60 % |

Le réseau plus profond apprend donc moins rapidement.

Il ne s'agit pas d'un sur-apprentissage, car les performances sur le train sont elles-mêmes moins bonnes que celles d'AlexNet standard.

On observe plutôt une difficulté d'optimisation, donc un comportement de **sous-apprentissage au début de l'entraînement**.

En ajoutant des couches, le gradient doit traverser davantage d'opérations pendant la rétropropagation. Il peut donc s'atténuer avant d'atteindre les premières couches, ce qui ralentit leur mise à jour et rend le démarrage de l'apprentissage plus difficile.

---

# Exercice 4

AlexNet a ensuite été entraîné sur l'intégralité du jeu d'entraînement ImageNet avec :

```text
subset_size = None
batch_size  = 256
GPU         = NVIDIA L4
```

Le job a été arrêté après **8 époques complètes**, ce qui est suffisant pour réaliser l'analyse demandée.

---

## 1. Profilage temporel

Les temps observés sont :

| Époque | Temps |
| ---: | ---: |
| 1 | 24 min 06 s |
| 2 | 24 min 04 s |
| 3 | 24 min 06 s |
| 4 | 24 min 06 s |
| 5 | 24 min 02 s |
| 6 | 24 min 02 s |
| 7 | 24 min 05 s |
| 8 | 23 min 54 s |

La première époque complète a donc pris exactement :

```text
24 min 06 s
```

Sur les huit époques, la moyenne est d'environ :

```text
24 min 03 s par époque
```

À cette vitesse, pour 100 époques :

```math
100 \times 24.052
=
2405.2\ \text{minutes}
```

soit environ :

```math
\boxed{40.1\ \text{heures}}
```

c'est-à-dire approximativement :

```text
40 h 05 min
```

sans Early Stopping.

---

## 2. Évolution sur ImageNet complet

Les valeurs obtenues sont :

| Époque | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.6903 | 4.49 % | 4.7430 | 11.34 % |
| 2 | 4.7358 | 12.12 % | 4.1073 | 18.78 % |
| 3 | 4.3138 | 17.12 % | 3.7226 | 24.23 % |
| 4 | 4.0356 | 20.76 % | 3.4710 | 28.14 % |
| 5 | 3.8368 | 23.51 % | 3.2997 | 30.88 % |
| 6 | 3.6852 | 25.73 % | 3.1865 | 32.88 % |
| 7 | 3.5627 | 27.52 % | 3.0588 | 34.58 % |
| 8 | 3.4643 | 29.06 % | 2.9512 | 36.60 % |

Les deux losses diminuent régulièrement tandis que les accuracies augmentent.

À l'époque 8, on obtient :

```text
Train Accuracy = 29.06 %
Val Accuracy   = 36.60 %
```

L'apprentissage est donc encore clairement en progression.

---

### Comparaison avec le sous-ensemble de 20 000 images

#### Loss

![Loss AlexNet complet vs sous-ensemble](figures/TensorBoard_Loss_AlexNet_Full_vs_Subset.png)

#### Accuracy

![Accuracy AlexNet complet vs sous-ensemble](figures/TensorBoard_Accuracy_AlexNet_Full_vs_Subset.png)

Sur le sous-ensemble de 20 000 images, à l'époque 10 :

```text
Train Loss = 6.4079
Val Loss   = 6.4746
```

On commence à avoir :

```math
\text{Loss}_{val} > \text{Loss}_{train}
```

avec un petit écart entre train et validation.

Sur ImageNet complet, par exemple à l'époque 8 :

```text
Train Loss = 3.4643
Val Loss   = 2.9512
```

Le sur-apprentissage est beaucoup moins visible sur le dataset complet.

La raison principale est la quantité de données : AlexNet dispose d'environ **1,28 million d'images d'entraînement** au lieu de seulement 20 000.

Avec beaucoup plus d'exemples différents, il est plus difficile pour le réseau de simplement mémoriser les données d'entraînement.

L'augmentation de données et le Dropout limitent également le sur-apprentissage.

---

## 3. Pourquoi la `Loss/Validation` est-elle inférieure à la `Loss/Train` ?

Nous observons par exemple à l'époque 1 :

```text
Train Loss = 5.6903
Val Loss   = 4.7430
```

Deux éléments du code expliquent principalement ce résultat.

### Dropout

Pendant le train :

```python
model.train()
```

le Dropout est actif.

Une partie des activations est donc supprimée aléatoirement, ce qui rend volontairement l'apprentissage plus difficile.

Pendant la validation :

```python
model.eval()
```

le Dropout est désactivé et le réseau utilise toutes ses activations.

La loss de validation peut donc être plus faible.

### Data Augmentation

Pendant l'entraînement, les images subissent :

```python
RandomResizedCrop(227)
RandomHorizontalFlip()
```

Les images vues par le réseau sont donc modifiées aléatoirement et peuvent être plus difficiles à classifier.

La validation utilise au contraire :

```python
Resize(256)
CenterCrop(227)
```

de façon déterministe.

Ces deux différences expliquent pourquoi on peut observer au début :

```math
\text{Loss}_{Validation}
<
\text{Loss}_{Train}
```

sans qu'il s'agisse d'une erreur.

---

# Bilan

Sur les expériences réalisées :

- AlexNet apprend mieux que le MLP sur les images ;
- augmenter simplement la profondeur d'AlexNet rend ici le début de l'optimisation plus difficile ;
- sur ImageNet complet, les performances progressent fortement et le sur-apprentissage est beaucoup moins visible que sur le sous-ensemble de 20 000 images ;
- une époque d'entraînement sur ImageNet complet prend environ `24 min 03 s` sur la NVIDIA L4.