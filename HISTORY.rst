=======
History
=======

0.6 (2020-11-20)
----------------

* Distortion is now performed on the fly
* Removed unused tiff recipe

0.5 (2020-11-02)
----------------

* Remove preprocess and update for new metadata representation

0.4.4 (2020-06-19)
------------------

* Modify chunk size to run smaller number of tasks
* Clean temporary files
* Perform distortion correction per z slice

0.4.1 (2020-06-10)
------------------

* Improved performance of downsample step
* Fix multi-slice registration memory consumption
 
0.4.0 (2020-06-09)
------------------

* Improve performance of distortion correction
* Improve performace of mosaic creation
* Add multi-slice registration

0.3.0 (2020-02-20)
------------------

* Improved offset calculation between tiles.
* Improved flatfield and dark correction.
* Use XArray/Zarr as internal data format.
* Output multi-resolution XArray/Zarr datasets.
* Output multi-resolution TIFF images.
* Update documentation.

0.2.1 (2019-11-19)
------------------

* First release.
