# ImageNetPlaces365 (Rescaling large datasets)
Abstract: The fact that several categories in a large dataset are not difficult to be recognized by
deep neural networks, eliminating them for a challenging tiny set will assist that the early
proposals of networks can take a quick trial of verification. To this end, we propose
an efficient rescaling method based on the validation outcomes of a pre-trained model.
Firstly, it will take out the sensitive images of the lowest-accuracy classes of the validation
outcomes. Each of such images is then considered to identity which label it was
confused with. Experimental results for rescaling two popular large datasets (ImageNet
and Places365) have proved that gathering the lowest-accuracy classes along with the
most confused ones can product small subsets with more challenge for a quick validation
of an early network draft. Utilizing these rescaled sets will help researchers to save time
and computational cost in the way of designing deep neural architectures.
* Efficiently rescaling a large dataset by adapting statistical
computation to validation results of a pre-trained network.
* A unified collection of the sensitive images and those in
their confused classes would form a challenging tiny set.
* An application to rescaling two large datasets: ImageNet
and Places365 to obtain their rescaled subsets.
* Experimental results for image classification have validated
the raised challenge of the rescaled subsets. Verifying on
these subsets helps researchers to save the computational
cost for the early network drafts.
* It can be conducive to a good rate of a network draft on the
large datasets if it is good on the rescaled subsets.
