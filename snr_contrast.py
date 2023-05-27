import cv2
import numpy as np

"""
def pre_process(img, filter, k): # k: kernel
    img = cv2.imread(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if filter=='gaussian':
        new_img = cv2.GaussianBlur(img, (k, k), 0)
    if filter=='median':
        new_img = cv2.medianBlur(img, k)
    #if filter=='bilateral':
    #    new_img = cv2.bilateralFilter(img, 10, 100, 100)
    if filter=='unsharp':
        img2 = cv2.GaussianBlur(img, (k, k), 0)
        img3 = img - img2  # mask
        new_img = img + img3
    return new_img
"""


def getPSNR(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    s1 = np.abs(img1 - img2)  # |I1 - I2|
    s1 = np.float32(s1)  # cannot make a square on 8 bits
    s1 = s1 * s1  # |I1 - I2|^2
    sse = s1.sum()  # sum elements per channel
    if sse <= 1e-10:  # sum channels
        return 0  # for small values return zero
    else:
        shape = img1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 ** 2) / mse)
        print('PSNR:', psnr)
        return psnr


img1 = '/home/danny/Code/Mosquinha/Melanogaster_after.png'
img2 = ''
"""
img1 = '/home/danny/Code/images_legs/americana_norte.png'
img2 = '/home/danny/Code/images_legs/americana_norte_no_back.png'
cv2.imread(img2)
img = pre_process(img2, 'gaussian', 5)
cv2.imwrite('/home/danny/Code/images_legs/americana_norte_no_back.png', img)
img_filtered = '/home/danny/Code/images_legs/americana_norte_no_back.png'
getPSNR(img1, img_filtered)
"""

"""
#no_back = '/home/danny/Code/Mosquinha/best3_melanogaster.png'
filter = pre_process(img, 'unsharp', 5)
cv2.imwrite('/home/danny/Code/Mosquinha/melanogaster3_unsharp_5.png', filter)
filter = '/home/danny/Code/Mosquinha/melanogaster3_unsharp_5.png'
getPSNR(img, filter)
"""


def getContrast(img):
    img = cv2.imread(img)
    maximum = np.max(img)
    print(maximum)
    minimum = np.min(img)
    print(minimum)
    # contrast = (maximum-minimum)/(maximum+minimum)
    # if maximum != minimum:
    #    contrast = (maximum - minimum)/(maximum + minimum)
    # else:
    contrast = maximum - minimum
    print("Constrast is of ", contrast)


"""
img1 = '/home/danny/Code/images_legs/americana_norte.png'
getContrast(img1)
img2 = '/home/danny/Code/images_legs/americana_norte_no_back.png'
getContrast(img2)
"""

"""
img = '/home/danny/Code/Mosquinha/best_melanogaster.png'
#no_back = '/home/danny/Code/Mosquinha/best3_melanogaster.png'
filter = pre_process(img, 'gaussian', 3)
cv2.imwrite('/home/danny/Code/Mosquinha/melanogaster_gaussian3.png', filter)
filter = '/home/danny/Code/Mosquinha/melanogaster_gaussian3.png'
getContrast(img)
"""
# WITHOUT BACKGROUND 

# Melanogaster

# n = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

"""
psnr = []
cont_before = []
cont_after = []
for i in n:
    original_img = cv2.imread('/home/danny/Code/Mosquinha/best2_melanogaster.png')
    print(type(original_img))
    filtered_img = pre_process(original_img, 'gaussian', i)
    filtered_img = cv2.imread('/home/danny/Code/gaussian.png')
    psnr_melanogaster = getPSNR(original_img, filtered_img)
    psnr.append(psnr_melanogaster)
    print("PSNR of the image melanogaster gaussian is", psnr_melanogaster)
    #contrast_before = getContrast('/home/danny/Code/Mosquinha/best_nova.png')
    #contrast_before = getContrast(original_img)
    #cont_before.append(contrast_before)
    #print('contrast before filtering', contrast_before)
    #contrast_after = getContrast('/home/danny/Code/gaussian.png')
    contrast_after = getContrast(filtered_img)
    cont_after.append(contrast_after)
    print('contrast after filtering', contrast_after)
    print('psnr[i]', psnr)
plt.plot(n, psnr)
plt.xlabel("Kernel size")
plt.ylabel("PSNR (dB)")
#plt.plot(psnr)
#plt.plot(cont_before, cont_after)
#plt.title("Contrast variation for Melanogaster species with unsharp masking")
#plt.ylabel('Contrast')
#plt.xlabel('Kernel size')
plt.show()
"""

"""
def main():

    original_img = cv2.imread('/home/danny/Code/Mosquinha/best_melanogaster.png')
    filtered_img = pre_process(original_img, 'gaussian')
    filtered_img = cv2.imread('/home/danny/Code/gaussian.png')
    psnr_melanogaster = getPSNR('/home/danny/Code/Mosquinha/best_melanogaster.png', '/home/danny/Code/gaussian.png')
    print("PSNR of the image melanogaster gaussian is", psnr_melanogaster)
    contrast_before = getContrast('/home/danny/Code/Mosquinha/best_nova.png')
    print('contrast before filtering', contrast_before)
    contrast_after = getContrast('/home/danny/Code/gaussian.png')
    print('contrast after filtering', contrast_after)

    print('\n \n')

    original_img = cv2.imread('/home/danny/Code/Mosquinha/best_melanogaster.png')
    filtered_img = pre_process(original_img, 'median')
    filtered_img = cv2.imread('/home/danny/Code/median.png')
    psnr_melanogaster = getPSNR('/home/danny/Code/Mosquinha/best_melanogaster.png', '/home/danny/Code/median.png')
    print("PSNR of the image melanogaster gaussian is", psnr_melanogaster)
    contrast_before = getContrast('/home/danny/Code/Mosquinha/best_melanogaster.png')
    print('contrast before filtering', contrast_before)
    contrast_after = getContrast('/home/danny/Code/median.png')
    print('contrast after filtering', contrast_after)

    print('\n \n')

    original_img = cv2.imread('/home/danny/Code/Mosquinha/best_melanogaster.png')
    filtered_img = pre_process(original_img, 'bilateral')
    filtered_img = cv2.imread('/home/danny/Code/bilateral.png')
    psnr_melanogaster = getPSNR('/home/danny/Code/Mosquinha/best_melanogaster.png', '/home/danny/Code/bilateral.png')
    print("PSNR of the image melanogaster gaussian is", psnr_melanogaster)
    contrast_before = getContrast('/home/danny/Code/Mosquinha/best_melanogaster.png')
    print('contrast before filtering', contrast_before)
    contrast_after = getContrast('/home/danny/Code/bilateral.png')
    print('contrast after filtering', contrast_after)

    print('\n \n')

    original_img = cv2.imread('/home/danny/Code/Mosquinha/best_melanogaster.png')
    filtered_img = pre_process(original_img, 'unsharp')
    filtered_img = cv2.imread('/home/danny/Code/unsharp.png')
    psnr_melanogaster = getPSNR('/home/danny/Code/Mosquinha/best_melanogaster.png', '/home/danny/Code/unsharp.png')
    print("PSNR of the image melanogaster unsharp is", psnr_melanogaster)
    contrast_before = getContrast('/home/danny/Code/Mosquinha/best_melanogaster.png')
    print('contrast before filtering', contrast_before)
    contrast_after = getContrast('/home/danny/Code/unsharp.png')
    print('contrast after filtering', contrast_after)
"""
"""
    n = [3, 5, 7, 9, 15]
    psnr = []
    for i in n:
        original_img = cv2.imread('/home/danny/Code/Mosquinha/best_melanogaster.png')
        filtered_img = pre_process(original_img, 'gaussian')
        filtered_img = cv2.imread('/home/danny/Code/gaussian.png')
        psnr_melanogaster = getPSNR('/home/danny/Code/Mosquinha/best_melanogaster.png', '/home/danny/Code/gaussian.png')
        psnr.append(psnr_melanogaster)
        print("PSNR of the image melanogaster gaussian is", psnr_melanogaster)
        contrast_before = getContrast('/home/danny/Code/Mosquinha/best_nova.png')
        print('contrast before filtering', contrast_before)
        contrast_after = getContrast('/home/danny/Code/gaussian.png')
        print('contrast after filtering', contrast_after)
        return psnr
"""

# if __name__ == '__main__':
# main()
