import matplotlib
from cycler import cycler

matplotlib.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
matplotlib.rcParams['animation.writer'] = 'ffmpeg'
matplotlib.rcParams['animation.html'] = 'html5'
matplotlib.rcParams['figure.figsize'] = (7.0, 5.0)


matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
                                                ['#2F528F', '#FFC000', '#843C0C', '#ef5675', '#d97642', '#8c564b',
                                                 '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri"],
    "font.size": 20,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.labelweight": "bold",
    "axes.labelsize": 24,
    "axes.linewidth": "2"
})

brains_to_silhouette = {
    'Deer': 'https://www.shutterstock.com/image-vector/white-tailed-buck-deer-silhouette-600w-2388718177.jpg',
    'Koala': 'https://www.shutterstock.com/image-vector/cute-vector-koala-sleeping-on-600w-1608080200.jpg',
    'Giraffe': 'https://www.shutterstock.com/image-vector/vector-drawing-silhouette-giraffe-ink-600w-1563230653.jpg',
    'Rat': 'https://www.shutterstock.com/image-vector/black-silhouette-rat-mouse-sitting-600w-519807670.jpg',
    'Cynnopterus': 'https://www.shutterstock.com/image-vector/halloween-bat-silhouette-vector-design-600w-743323486.jpg',
    'Gerbil': 'https://www.shutterstock.com/image-vector/mongolian-gerbil-meriones-unguiculatus-silhouette-600w-1681322530.jpg',
    'Otter': 'https://www.shutterstock.com/image-vector/vector-silhouette-sitting-otter-600w-630966062.jpg',
    'BrownLemur': 'https://www.shutterstock.com/image-vector/best-lemur-silhouette-images-this-600w-2133620371.jpg',
    'Orangutan': 'https://www.shutterstock.com/image-vector/gorilla-silhouette-icon-illustration-template-600w-2205361665.jpg',
    'Macaque': 'https://www.shutterstock.com/image-vector/silhouette-monkey-animal-genus-primates-600w-1559630309.jpg',
    'HowlerMonkey': 'https://www.shutterstock.com/shutterstock/photos/1815244919/display_1500/stock-vector-standing-howler-monkey-alouatta-side-view-silhouette-found-in-map-central-and-south-america-1815244919.jpg',
    'Human': 'https://www.shutterstock.com/image-vector/stock-vector-featureless-human-figure-half-body-monochrome-illustration-or-male-skinhead-monochrome-silhouette-600w-2641858453.jpg',
    'Dog': 'https://www.shutterstock.com/image-vector/stock-vector-sitting-labrador-retriever-vector-silhouette-600w-2387616261.jpg',
    'FruitBat': 'https://www.shutterstock.com/image-vector/stock-vector-sleeping-bat-on-the-branch-tree-silhouette-for-halloween-poster-art-illustration-horror-movie-or-600w-2257463371.jpg'
}
