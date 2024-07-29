import json
import os

train_split_scene_idx_list = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 138, 139, 149, 150, 151, 152, 154, 155, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 187, 188, 190, 191, 192, 193, 194, 195, 196, 199, 200, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 218, 219, 220, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 315, 316, 317, 318, 321, 323, 324, 328, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 405, 406, 407, 408, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 464, 465, 467, 468, 469, 471, 472, 474, 475, 476, 477, 478, 479, 480, 499, 500, 501, 502, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 517, 518, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 541, 542, 543, 544, 545, 546, 566, 568, 570, 571, 572, 573, 574, 575, 576, 577, 578, 580, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 681, 683, 684, 685, 686, 687, 688, 689, 695, 696, 697, 698, 700, 701, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 726, 727, 728, 730, 731, 733, 734, 735, 736, 737, 738, 739, 740, 741, 744, 746, 747, 749, 750, 751, 752, 757, 758, 759, 760, 761, 762, 763, 764, 765, 767, 768, 769, 786, 787, 789, 790, 791, 792, 803, 804, 805, 806, 808, 809, 810, 811, 812, 813, 815, 816, 817, 819, 820, 821, 822, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 858, 860, 861, 862, 863, 864, 865, 866, 868, 869, 870, 871, 872, 873, 875, 876, 877, 878, 880, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 945, 947, 949, 952, 953, 955, 956, 957, 958, 959, 960, 961, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 988, 989, 990, 991, 992, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1104, 1105, 1106, 1107, 1108, 1109, 1110]

val_split_scene_idx_list = [3, 12, 13, 14, 15, 16, 17, 18, 35, 36, 38, 39, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 221, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 329, 330, 331, 332, 344, 345, 346, 519, 520, 521, 522, 523, 524, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 625, 626, 627, 629, 630, 632, 633, 634, 635, 636, 637, 638, 770, 771, 775, 777, 778, 780, 781, 782, 783, 784, 794, 795, 796, 797, 798, 799, 800, 802, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 962, 963, 966, 967, 968, 969, 971, 972, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073]

train_data_scene_size_list = [40, 40, 40, 39, 40, 39, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 39, 40, 41, 39, 40, 39, 39, 40, 40, 39, 39, 39, 40, 39, 39, 40, 41, 39, 39, 41, 40, 39, 39, 39, 40, 40, 39, 40, 39, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 39, 40, 39, 39, 39, 40, 40, 39, 40, 40, 40, 40, 39, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 39, 40, 39, 40, 40, 40, 40, 40, 39, 32, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 40, 39, 39, 40, 40, 39, 40, 39, 40, 39, 39, 40, 39, 40, 40, 39, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 39, 40, 39, 40, 40, 39, 40, 40, 40, 40, 39, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 39, 40, 40, 39, 40, 40, 39, 39, 40, 40, 40, 40, 40, 39, 40, 40, 39, 39, 39, 39, 40, 40, 40, 39, 39, 40, 39, 39, 40, 40, 39, 39, 39, 39, 41, 40, 40, 40, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 40, 41, 40, 40, 39, 41, 40, 40, 40, 40, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 40, 40, 40, 41, 40, 40, 40, 41, 40, 41, 41, 41, 41, 40, 41, 40, 41, 41, 41, 41, 40, 41, 41, 41, 41, 41, 41, 40, 41, 41, 40, 41, 41, 40, 40, 40, 41, 41, 41, 40, 41, 41, 41, 40, 41, 40, 41, 41, 40, 40, 40, 41, 40, 41, 41, 41, 41, 41, 41, 41, 40, 41, 41, 40, 41, 40, 40, 40, 40, 41, 41, 41, 40, 41, 40, 40, 40, 39, 41, 39, 40, 40, 41, 41, 40, 41, 41, 40, 40, 40, 41, 40, 41, 40, 41, 41, 40, 41, 41, 41, 40, 40, 41, 41, 40, 41, 41, 41, 41, 40, 40, 41, 41, 41, 41, 41, 40, 41, 41, 41, 40, 40, 40, 41, 41, 41, 40, 41, 40, 41, 41, 41, 40, 41, 40, 41, 41, 40, 40, 41, 41, 40, 41, 40, 41, 40, 40, 40, 41, 41, 40, 41, 40, 41, 40, 41, 40, 41, 41, 41, 40, 41, 40, 41, 40, 40, 40, 40, 40, 40, 41, 41, 41, 40, 41, 41, 41, 41, 41, 40, 41, 40, 40, 41, 41, 41, 40, 40, 41, 41, 41, 41, 41, 40, 40, 40, 40, 41, 40, 41, 40, 40, 40, 41, 40, 40, 41, 40, 41, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 40, 40, 40, 41, 40, 41, 41, 41, 41, 40, 41, 41, 41, 41, 41, 41, 41, 40, 40, 41, 41, 40, 41, 40, 41, 40, 41, 41, 40, 40, 40, 40, 40, 40, 40, 41, 40, 41, 40, 40, 40, 40, 41, 40, 40, 41, 40, 41, 40, 40, 40, 40, 41, 40, 41, 40, 41, 41, 40, 41, 41, 40, 41, 41, 40, 41, 41, 41, 41, 41, 41, 40, 41, 41, 40, 40, 40, 39, 40, 41, 40, 41, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 40, 41, 40, 40, 40, 40, 40, 41, 40, 40, 41, 40, 40, 40, 40, 40, 40, 41, 40, 41, 40, 41, 41, 40, 40, 41, 40, 41, 41, 41, 40, 40, 41, 40, 41, 41, 41, 40, 40, 40, 41, 40, 40, 41, 41, 40, 41, 41, 40, 41, 40, 40, 41, 40, 40, 40, 40, 41, 40, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 41, 41, 40, 41, 40, 40, 40, 40, 40, 41, 40, 41, 40, 41, 40, 40, 40, 41, 41, 40, 40, 40, 40, 40, 40, 41, 41, 40, 40, 40, 40, 40, 41, 41, 41, 40, 40, 41, 40, 40, 40, 40, 40, 40, 41, 40, 40, 41, 40, 39, 40]

sorted_scene_idx_list = [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 199, 200, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 315, 316, 317, 318, 177, 178, 179, 180, 181, 182, 183, 184, 185, 187, 188, 218, 220, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 190, 191, 192, 193, 194, 195, 196, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 138, 139, 149, 150, 151, 152, 154, 155, 157, 158, 159, 160, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 382, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 375, 373, 374, 376, 377, 378, 379, 380, 381, 383, 384, 385, 386, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 328, 499, 500, 501, 502, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 665, 666, 667, 668, 669, 670, 671, 672, 673, 321, 674, 675, 676, 677, 678, 679, 681, 683, 684, 685, 686, 323, 687, 688, 689, 324, 514, 515, 517, 518, 695, 696, 697, 698, 700, 701, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 726, 727, 728, 730, 731, 733, 734, 735, 736, 737, 738, 739, 740, 741, 744, 746, 747, 749, 750, 751, 752, 757, 758, 759, 760, 761, 762, 763, 764, 765, 767, 768, 769, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 541, 542, 543, 544, 545, 546, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 464, 465, 467, 468, 469, 471, 472, 474, 475, 476, 477, 478, 479, 480, 566, 568, 570, 571, 572, 573, 574, 575, 576, 577, 578, 580, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 868, 869, 870, 871, 872, 873, 875, 876, 877, 878, 880, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 803, 804, 805, 806, 808, 809, 810, 811, 812, 813, 815, 816, 817, 819, 820, 821, 822, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 988, 989, 990, 991, 945, 947, 949, 952, 953, 955, 956, 957, 958, 959, 960, 961, 399, 400, 401, 402, 403, 405, 406, 407, 408, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 786, 787, 789, 790, 791, 792, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 858, 860, 861, 862, 863, 864, 865, 866, 992, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1104, 1105, 1106, 1107, 1108, 1109, 1110]

new_train_data_scene_size_list = [40, 40, 40, 39, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 39, 40, 39, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 39, 40, 41, 39, 40, 39, 39, 40, 40, 39, 39, 39, 40, 39, 39, 40, 41, 39, 39, 41, 40, 39, 40, 39, 39, 40, 39, 39, 40, 40, 39, 40, 39, 40, 39, 39, 39, 39, 39, 39, 40, 39, 40, 39, 40, 40, 40, 40, 40, 39, 32, 39, 40,  40, 40, 39, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 39, 40, 39, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 39, 40, 39, 39, 39, 40, 40, 39, 40, 39, 39, 40, 40, 39, 40, 39, 40, 40, 40, 40, 40, 40, 39, 40, 39, 40, 39, 39, 40, 40, 39, 39, 39, 39, 41, 40, 40, 40, 40, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 40, 40, 39, 41, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 39, 40, 40, 40, 40, 40, 40, 41, 40, 41, 40, 41, 40, 40, 40, 40, 40, 40, 41, 39, 40, 40, 39, 41, 39, 40, 40, 41, 41, 40, 41, 41, 40, 40, 41, 41, 40, 41, 41, 41, 41, 41, 40, 40, 41, 40, 40, 41, 41, 41, 40, 40, 41, 41, 41, 40, 41, 41, 40, 40, 40, 41, 40, 41, 40, 40, 40, 41, 40, 41, 40, 40, 40, 41, 40, 40, 41, 40, 41, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 40, 40, 40, 41, 40, 41, 41, 41, 41, 40, 41, 41, 41, 41, 41, 41, 41, 40, 40, 41, 41, 40, 41, 40, 41, 40, 41, 41, 40, 41, 41, 40, 41, 41, 41, 40, 40, 41, 41, 40, 41, 41, 41, 41, 40, 40, 41, 41, 41, 40, 39, 40, 40, 40, 40, 40, 40, 39, 40, 40, 39, 40, 40, 39, 39, 40, 40, 40, 40, 40, 39, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 40, 41, 41, 41, 40, 41, 40, 41, 41, 40, 40, 40, 41, 40, 41, 41, 41, 41, 41, 41, 41, 40, 41, 41, 40, 41, 40, 40, 40, 40, 41, 41, 41, 40, 41, 40, 41, 41, 40, 41, 41, 41, 40, 40, 40, 41, 41, 41, 40, 41, 40, 41, 41, 41, 40, 41, 40, 41, 41, 40, 40, 41, 41, 40, 41, 40, 41, 40, 40, 40, 41, 41, 40, 41, 40, 41, 40, 41, 40, 41, 41, 41, 41, 40, 41, 41, 40, 40, 40, 39, 40, 41, 40, 41, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 40, 41, 40, 40, 40, 40, 40, 41, 40, 40, 40, 41, 40, 41, 40, 40, 40, 40, 41, 40, 40, 41, 40, 41, 40, 40, 40, 41, 40, 40, 41, 40, 41, 41, 41, 40, 40, 41, 40, 41, 41, 41, 40, 40, 40, 40, 40, 40, 41, 40, 41, 40, 41, 41, 40, 40, 40, 41, 40, 40, 40, 41, 40, 41, 41, 41, 41, 40, 41, 40, 41, 41, 40, 40, 40, 40, 40, 40, 41, 41, 40, 41, 41, 41, 41, 41, 41, 40, 41, 41, 40, 41, 41, 40, 40, 40, 41, 41, 40, 41, 40, 41, 40, 41, 41, 40, 41, 41, 40, 41, 41, 40, 41, 41, 41, 41, 41, 40, 40, 40, 41, 40, 40, 41, 41, 40, 41, 41, 40, 41, 40, 40, 41, 40, 40, 40, 40, 41, 40, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 41, 41, 40, 41, 40, 40, 40, 40, 40, 41, 40, 41, 40, 41, 40, 40, 40, 41, 41, 40, 40, 40, 40, 40, 40, 41, 41, 40, 40, 40, 40, 40, 41, 41, 41, 40, 40, 41, 40, 40, 40, 40, 40, 40, 41, 40, 40, 41, 40, 39, 40]

sum = 0
a_dict = {}
for i in range(len(new_train_data_scene_size_list)):
    a_dict[i] = list(range(sum, sum + new_train_data_scene_size_list[i]))
    sum += new_train_data_scene_size_list[i]

new_group_idx_to_sample_idxs = a_dict
