# A propos de du script detection_ellipse.py

## step 1:  chargement de l'image

![alt text](https://github.com/TestaVuota/ImageAnalysis/blob/main/WilliamIm/images/dimer_i003.jpg?raw=true)

<!-- ## step 2:  application d'un filtre/blur gaussien σ=2 -->
## step 2:  application d'un filtre/blur gaussien σ=1
![alt text](https://github.com/TestaVuota/ImageAnalysis/blob/main/WilliamIm/images/filtered_gaussian.png?raw=true)

## step 3:  application du [rolling ball algorithm](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rolling_ball.html)
![alt text](https://github.com/TestaVuota/ImageAnalysis/blob/main/WilliamIm/images/rolling_ball.png?raw=true)

## step 4:  deduction du mask resultant et application du mask sur l'image d'origine via plotly
<!-- ![alt text](https://github.com/TestaVuota/ImageAnalysis/blob/main/WilliamIm/deducedMasks.png?raw=true) -->
![alt text](https://github.com/TestaVuota/ImageAnalysis/blob/main/WilliamIm/images/plotly.png?raw=true)


# A faire:  

- idée 1: deduction d'arc de cercle:
    - cad revoir le edge detector
    - fitter avec un polynome de degré 2 
    - en déduire le rayon de coubure & centre associée
    - déterminer une distance minimal entre deux centres calculée en vu de faire une association de centre 
    - reconstruction d'un cercle/ellipse de centre: 'moyenne des deux centres d'arc de cercle' et de rayon: 'moyenne des deux rayons de courbure'

- idée 2: Voir l'effet d'une transformée de hough sur l' image:
    - permettrait de voir les sommets ou points de 'concentration' de l'image
    - et ainsi tracer des formes -> revoir le fill par triangle de [Delaunay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html), voir [lien](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.find_simplex.html) ( pour forme la détection de forme triangulaire)
    - revoir la fonction trace polynome en fonction du nombre de sommet (module (shapely?)[https://shapely.readthedocs.io/en/latest/manual.html#polygons] depend de [GEOS?](https://geos.readthedocs.io/en/latest/users.html))
        - avec [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) & [version spatial delaynay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.plane_distance.html)
        - avec [shapely](https://stackoverflow.com/questions/30457089/how-to-create-a-shapely-polygon-from-a-list-of-shapely-points)
        - rien à voir 
        [polygonal tracing](https://learn.microsoft.com/en-us/dotnet/api/microsoft.azure.documents.spatial.polygon.-ctor?view=azure-dotnet)
        (avec [blender](https://blender.stackexchange.com/questions/102597/finding-vertices-edges-faces-and-tris-using-python))

- idée 3: Revoir l'algorythme de rolling_ball en vue de le modifier pour d'autre forme mais aussi:
    - le comparer avec sa forme analogue trouvée sur opencv-python (cf. murtaza)
    - voir l'analyse utilisée (il me semble que c'était une transformée de hough)

-idée 4: suivre tuto filtering image
    - [lien](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0205736)

<!-- # draft

- [add images in .md](https://fr.code-paper.com/shell-bash/examples-how-to-add-images-in-md-files)
- [add images in .md](https://www.digitalocean.com/community/tutorials/markdown-markdown-images) -->