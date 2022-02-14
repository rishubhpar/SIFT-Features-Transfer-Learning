import cv2
import numpy as np
import math

# Computing the local maxima within a region
def compute_local_maxima(dogs, id, x, y):
    top_scale = dogs[id-1, x-1:x+2:, y-1:y+2].ravel()
    mid_scale = dogs[id, x-1:x+2:, y-1:y+2].ravel()
    mid_scale[4] = -1 # Supressing the central pixel 
    bottom_scale = dogs[id+1, x-1:x+2, y-1:y+2].ravel()

    neighs = np.concatenate([top_scale, mid_scale, bottom_scale])
    # print("nums: ", neighs.shape, dogs[id, x, y])

    max = np.max(neighs)
    if (dogs[id, x, y] == max):
        # print("neighs: ", neighs)
        return True
    
    return False 


# Compute scale space with dog
def compute_scale_space(img, s):
    img_disp = img.copy()
    k_size = 15

    # Pre prop
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 3)  

    k = math.pow(2.0, (1/s))
    n = img.shape[0]
    m = img.shape[1]

    # Computing the gaussians
    gaussians = []
    for id in range(0, s+1):
        sigma = math.pow(k, id)
        gauss = cv2.GaussianBlur(img, (k_size,k_size), sigma)
        gaussians.append(gauss)

    # Computing the difference of Gaussians 
    dogs = []
    for id in range(0, s):
        diff = gaussians[id+1] - gaussians[id]
        # print("diff: ", diff)
        dogs.append(diff)

    dogs = np.array(dogs)
    # print("dogs shape: ", dogs.shape)  

    keypoint_matrix = np.zeros(img.shape)

    # Computing the difference between scales 
    for id in range(1, s-1):
        for x in range(1,n-1):
            for y in range(1,m-1):
                is_max = compute_local_maxima(dogs, id, x, y) 
                # print("is max: ", is_max)
                if (is_max):
                    keypoint_matrix[x,y] = 1
                    img_disp[x, y, :] = [0, 0, 255] 



    return img_disp


def compute_sift():
    img_list = ['cube1.jpeg', 'cube2.png', 'input1.jpeg', 'input2.jpeg', 'input3.png', 'india_gt1.jpeg', 'india_gt3.jpeg']

    print("Processing")
    for id in range(0, len(img_list)):
        img_path = './figures/' + img_list[id]
        dst_path = './figures/' + img_list[id][:-5] + '_rotated_kp.jpg'
        s = 6

        print("img path: ", img_path)
        img = cv2.imread(img_path)
        h = 225
        w = (img.shape[1] / img.shape[0]) * h
        w = int(w)

        # Original Image
        # img = cv2.resize(img, (int(w), int(h))) 

        # Zoomed Image
        # img = cv2.resize(img, (5*w, 5*h))
        # img = img[1*h:3*h, 1*w:3*w, :]

        # Rotated Image
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), +45, 1.0)   
        img = cv2.warpAffine(img, M, (w, h))

        img_kp = compute_scale_space(img, s) 
        
        cv2.imwrite(dst_path, img_kp)
 

# Main function to call the sift module 
def run_main():
    compute_sift()


if __name__ == "__main__":
    run_main()