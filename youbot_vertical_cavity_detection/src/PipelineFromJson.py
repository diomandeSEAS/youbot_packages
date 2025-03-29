import json
import cv2
import numpy as np

class PipelineFromJSON:
    """
    A class that loads a pipeline from JSON and applies each filter in sequence
    to a given cv_image (BGR format).
    """

    # Dictionary mapping filter names to their implementations
    FILTERS_AVAILABLE = {}

    def __init__(self, json_path):
        """
        Constructor loads the filter pipeline from the given JSON file.
        """
        with open(json_path, 'r') as f:
            self.pipeline = json.load(f)  # a list of { "name": "...", "params": {...} }

    def apply(self, cv_image):
        """
        Applies the loaded pipeline (in order) to the given cv_image.
        Returns the resulting image.
        """
        result = cv_image.copy()
        for step in self.pipeline:
            filter_name = step["name"]
            params = step["params"]
            # Look up the filter function
            filter_func = self.FILTERS_AVAILABLE.get(filter_name)
            if not filter_func:
                print(f"[Warning] Filter '{filter_name}' not found in FILTERS_AVAILABLE. Skipping.")
                continue
            result = filter_func(result, params)
        return result

    # ----------------------
    # Existing Filter Implementations
    # ----------------------
    @staticmethod
    def filter_rgb(img, params):
        if not params.get("enable",True):return img
        bMin,bMax=params["bMin"],params["bMax"]
        gMin,gMax=params["gMin"],params["gMax"]
        rMin,rMax=params["rMin"],params["rMax"]
        lower=np.array([bMin,gMin,rMin],dtype=np.uint8)
        upper=np.array([bMax,gMax,rMax],dtype=np.uint8)
        mask=cv2.inRange(img,lower,upper)
        return cv2.bitwise_and(img,img,mask=mask)

    @staticmethod
    def filter_hsv(img,params):
        if not params.get("enable",True):return img
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hMin,hMax=params["hMin"],params["hMax"]
        sMin,sMax=params["sMin"],params["sMax"]
        vMin,vMax=params["vMin"],params["vMax"]
        lower=np.array([hMin,sMin,vMin],dtype=np.uint8)
        upper=np.array([hMax,sMax,vMax],dtype=np.uint8)
        mask=cv2.inRange(hsv,lower,upper)
        return cv2.bitwise_and(img,img,mask=mask)
    
    @staticmethod
    def filter_hsl(img,params):
        if not params.get("enable",True):return img
        hsl=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        hMin,hMax=params["hMin"],params["hMax"]
        lMin,lMax=params["lMin"],params["lMax"]
        sMin,sMax=params["sMin"],params["sMax"]
        lower=np.array([hMin,lMin,sMin],dtype=np.uint8)
        upper=np.array([hMax,lMax,sMax],dtype=np.uint8)
        mask=cv2.inRange(hsl,lower,upper)
        return cv2.bitwise_and(img,img,mask=mask)

    @staticmethod
    def filter_morph_grad(img,params):
        if not params.get("enable",True):return img
        k=params["kernel"]
        kernel=np.ones((k,k),np.uint8)
        return cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)

    @staticmethod
    def filter_erode(img,params):
        if not params.get("enable",True):return img
        k=params["kernel"]
        kernel=np.ones((k,k),np.uint8)
        return cv2.erode(img,kernel,iterations=1)

    @staticmethod
    def filter_dilate(img,params):
        if not params.get("enable",True):return img
        k=params["kernel"]
        kernel=np.ones((k,k),np.uint8)
        return cv2.dilate(img,kernel,iterations=1)

    @staticmethod
    def filter_sobel_x(img,params):
        if not params.get("enable",True):return img
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=params["ksize"])
        abs_sobelx=cv2.convertScaleAbs(sobelx)
        return cv2.cvtColor(abs_sobelx,cv2.COLOR_GRAY2BGR)

    @staticmethod
    def filter_gaussian_blur(img,params):
        if not params.get("enable",True):return img
        k=params["kernel"]
        if k%2==0:k+=1
        return cv2.GaussianBlur(img,(k,k),0)

    @staticmethod
    def filter_hist_equalization(img,params):
        if not params.get("enable",True):return img
        ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        ycrcb[...,0]=cv2.equalizeHist(ycrcb[...,0])
        return cv2.cvtColor(ycrcb,cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def filter_clahe_gray(img,params):
        if not params.get("enable",True):return img
        c=params.get("clipLimit",2.0)
        t=params.get("tileGrid",8)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        clahe=cv2.createCLAHE(clipLimit=c,tileGridSize=(t,t))
        eq=clahe.apply(gray)
        return cv2.cvtColor(eq,cv2.COLOR_GRAY2BGR)

    @staticmethod
    def filter_gamma_correction(img,params):
        if not params.get("enable",True):return img
        gamma=params.get("gamma",1.0)
        lut=np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img,lut)

    @staticmethod
    def filter_clahe_ycrcb(img,params):
        if not params.get("enable",True):return img
        c=params.get("clipLimit",2.0)
        t=params.get("tileGrid",8)
        ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        clahe=cv2.createCLAHE(clipLimit=c,tileGridSize=(t,t))
        ycrcb[...,0]=clahe.apply(ycrcb[...,0])
        return cv2.cvtColor(ycrcb,cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def filter_contours(img, params):
        if not params.get("enable", True):
            return img, None  # Ensure we always return img and a valid x position

        # Parameters for contour filtering
        n = params.get("nLargestContours", 1)
        crit = params.get("largestCriteria", "arcLength")
        lclip = params.get("lclip", 0.1)
        rclip = params.get("rclip", 0.9)
        tclip = params.get("tclip", 0.1)
        bclip = params.get("bclip", 0.9)
        tol = params.get("tol", 0.01)
        drawAll = params.get("drawAllContours", True)
        drawFiltered = params.get("drawFiltered", True)

        # Image dimensions
        h, w = img.shape[:2]

        # Convert to grayscale and find contours
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all detected contours (if enabled)
        if drawAll:
            cv2.drawContours(img, contours, -1, (0, 255, 0), 4)

        if not contours:
            print("No cavity edge detected: no contours found")
            return img, None

        # Filter contours based on X-clipping bounds
        filtered = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            if (lclip + tol) * w < cX < (rclip - tol) * w:
                filtered.append(c)

        if not filtered:
            print("No cavity edge detected: filters removed all contours")
            return img, None

        # Sorting function based on criteria
        def contour_key(c):
            if crit == "area":
                return cv2.contourArea(c)
            elif crit == "boundingRect":
                _, _, ww, hh = cv2.boundingRect(c)
                return ww * hh
            elif crit == "verticalLength":
                _, _, _, hh = cv2.boundingRect(c)
                return hh
            else:  # Default to arc length
                return cv2.arcLength(c, False)

        # Sort contours and pick the top N
        filtered.sort(key=contour_key, reverse=True)
        top = filtered[:n]

        any_drawn = False
        avg_x = None  # Default value if no contour passes filtering

        # Process the selected contours
        for c in top:
            pts = []
            for pt in c:
                x, y = pt[0]
                if (lclip + tol) * w < x < (rclip - tol) * w and (tclip + tol) * h < y < (bclip - tol) * h:
                    pts.append(pt)

            if not pts:
                continue

            if drawFiltered:
                cv2.drawContours(img, [np.array(pts, dtype=np.int32)], -1, (0, 255, 255), 6)

            # Calculate and draw average x-position of filtered points
            arr = np.array(pts).squeeze()
            avg_x = int(np.mean(arr[:, 0]))  # This is the actual return value

            # Draw the computed line at avg_x
            cv2.line(img, (avg_x, 0), (avg_x, h - 1), (0, 255, 255), 6)

            any_drawn = True

        if not any_drawn:
            print("No cavity edge detected after boundary check")
            return img, None

        return img, avg_x  # Correct return format

    @staticmethod
    def filter_simplify_colors(img,params):
        if not params.get("enable",True):return img
        K=params.get("K",5)
        attempts=params.get("attempts",10)
        max_iter=params.get("maxIter",100)
        epsilon=params.get("epsilon",0.2)
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        px=rgb.reshape(-1,3)
        crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,max_iter,epsilon)
        _,labels,centers=cv2.kmeans(np.float32(px),K,None,crit,attempts,cv2.KMEANS_PP_CENTERS)
        centers=np.uint8(centers)
        seg=centers[labels.flatten()]
        seg=seg.reshape(rgb.shape)
        seg_bgr=cv2.cvtColor(seg,cv2.COLOR_RGB2BGR)
        return seg_bgr

# Register filters in FILTERS_AVAILABLE
PipelineFromJSON.FILTERS_AVAILABLE = {
    "RGB":PipelineFromJSON.filter_rgb,
    "HSV":PipelineFromJSON.filter_hsv,
    "HSL":PipelineFromJSON.filter_hsl,
    "Morphological Gradient":PipelineFromJSON.filter_morph_grad,
    "Erode":PipelineFromJSON.filter_erode,
    "Dilate":PipelineFromJSON.filter_dilate,
    "SobelX":PipelineFromJSON.filter_sobel_x,
    "Contour Detection":PipelineFromJSON.filter_contours,
    "Gaussian Blur":PipelineFromJSON.filter_gaussian_blur,
    "Hist Equalization":PipelineFromJSON.filter_hist_equalization,
    "CLAHE (Gray)":PipelineFromJSON.filter_clahe_gray,
    "Gamma Correction":PipelineFromJSON.filter_gamma_correction,
    "CLAHE (YCrCb)":PipelineFromJSON.filter_clahe_ycrcb,
    "Color Simplify (K-Means)":PipelineFromJSON.filter_simplify_colors
}
