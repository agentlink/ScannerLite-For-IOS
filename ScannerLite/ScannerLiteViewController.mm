//
//  ScannerLiteViewController.m
//  ScannerLite
//
//  Created by link on 14-5-24.
//  Copyright (c) 2014年 link. All rights reserved.
//

#import "ScannerLiteViewController.h"
#include <opencv2/opencv.hpp>

static CGFloat DegreesToRadians(CGFloat degrees) {return degrees * M_PI / 180;};

@interface UIImage(UIImageScale)

-(UIImage*)scaleToSize:(CGSize)size;

-(UIImage*)getSubImage:(CGRect)rect;

@end

@implementation UIImage(UIImageScale)

//截取部分图像
-(UIImage*)getSubImage:(CGRect)rect
{
    CGImageRef subImageRef = CGImageCreateWithImageInRect(self.CGImage, rect);
    CGRect smallBounds = CGRectMake(0, 0, CGImageGetWidth(subImageRef), CGImageGetHeight(subImageRef));
    
    UIGraphicsBeginImageContext(smallBounds.size);
    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextDrawImage(context, smallBounds, subImageRef);
    UIImage* smallImage = [UIImage imageWithCGImage:subImageRef];
    UIGraphicsEndImageContext();
    CGImageRelease(subImageRef);
    
    return smallImage;
}

-(UIImage *)scaleToSize:(CGSize)targetSize
{
    UIImage *sourceImage = self;
    UIImage *newImage = nil;
    
    CGSize imageSize = sourceImage.size;
    CGFloat width = imageSize.width;
    CGFloat height = imageSize.height;
    
    CGFloat targetWidth = targetSize.width;
    CGFloat targetHeight = targetSize.height;
    
    CGFloat scaleFactor = 0.0;
    CGFloat scaledWidth = targetWidth;
    CGFloat scaledHeight = targetHeight;
    
    CGPoint thumbnailPoint = CGPointMake(0.0,0.0);
    
    if (CGSizeEqualToSize(imageSize, targetSize) == NO) {
        
        CGFloat widthFactor = targetWidth / width;
        CGFloat heightFactor = targetHeight / height;
        
        if (widthFactor < heightFactor)
            scaleFactor = widthFactor;
        else
            scaleFactor = heightFactor;
        
        scaledWidth  = width * scaleFactor;
        scaledHeight = height * scaleFactor;
        
        // center the image
        
        if (widthFactor < heightFactor) {
            thumbnailPoint.y = (targetHeight - scaledHeight) * 0.5;
        } else if (widthFactor > heightFactor) {
            thumbnailPoint.x = (targetWidth - scaledWidth) * 0.5;
        }
    }
    
    
    // this is actually the interesting part:
    
    UIGraphicsBeginImageContext(targetSize);
    
    CGRect thumbnailRect = CGRectZero;
    thumbnailRect.origin = thumbnailPoint;
    thumbnailRect.size.width  = scaledWidth;
    thumbnailRect.size.height = scaledHeight;
    
    [sourceImage drawInRect:thumbnailRect];
    
    newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    if(newImage == nil) NSLog(@"could not scale image");
    
    return newImage ;
}

- (UIImage *)imageRotatedByDegrees:(CGFloat)degrees
{
	// calculate the size of the rotated view's containing box for our drawing space
	UIView *rotatedViewBox = [[UIView alloc] initWithFrame:CGRectMake(0,0,self.size.width, self.size.height)];
	CGAffineTransform t = CGAffineTransformMakeRotation(DegreesToRadians(degrees));
	rotatedViewBox.transform = t;
	CGSize rotatedSize = rotatedViewBox.frame.size;
	
	// Create the bitmap context
	UIGraphicsBeginImageContext(rotatedSize);
	CGContextRef bitmap = UIGraphicsGetCurrentContext();
	
	// Move the origin to the middle of the image so we will rotate and scale around the center.
	CGContextTranslateCTM(bitmap, rotatedSize.width/2, rotatedSize.height/2);
	
	//   // Rotate the image context
	CGContextRotateCTM(bitmap, DegreesToRadians(degrees));
	
	// Now, draw the rotated/scaled image into the context
	CGContextScaleCTM(bitmap, 1.0, -1.0);
	CGContextDrawImage(bitmap, CGRectMake(-self.size.width / 2, -self.size.height / 2, self.size.width, self.size.height), [self CGImage]);
	
	UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
	UIGraphicsEndImageContext();
    
	return newImage;
}

@end

@interface ScannerLiteViewController ()

@end

@implementation ScannerLiteViewController

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    //CGColorSpaceRelease(colorSpace);
    
    return cvMat;
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

- (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    //CGColorSpaceRelease(colorSpace);
    
    return cvMat;
}

- (IplImage *)CreateIplImageFromUIImage:(UIImage *)image
{
    // Getting CGImage from UIImage
    CGImageRef imageRef = image.CGImage;
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    // Creating temporal IplImage for drawing
    IplImage *iplimage = cvCreateImage(cvSize(image.size.width,image.size.height), IPL_DEPTH_8U, 4);
    
    // Creating CGContext for temporal IplImage
    CGContextRef contextRef = CGBitmapContextCreate(iplimage->imageData, iplimage->width, iplimage->height,
                                                    iplimage->depth, iplimage->widthStep,
                                                    colorSpace, kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault);
    // Drawing CGImage to CGContext
    CGContextDrawImage(contextRef, CGRectMake(0, 0, image.size.width, image.size.height), imageRef);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    // Creating result IplImage
    IplImage *ret = cvCreateImage(cvGetSize(iplimage), IPL_DEPTH_8U, 3);
    cvCvtColor(iplimage, ret, CV_RGBA2BGR);
    cvReleaseImage(&iplimage);
    
    return ret;
}

- (UIImage *)CreateUIImageFromIplImage:(IplImage* )ipl_image
{
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    NSData* data = [NSData dataWithBytes: ipl_image->imageData length: ipl_image->imageSize];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef imageRef = CGImageCreate(ipl_image->width, ipl_image->height,
                                        ipl_image->depth, ipl_image->depth * ipl_image->nChannels, ipl_image->widthStep,
                                        colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault,
                                        provider, NULL, false, kCGRenderingIntentDefault);
    UIImage* ret = [UIImage imageWithCGImage: imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return ret;
}

-(UIImage *)CreateMosaicImage:(UIImage* )image
{
    cv::Mat src = [self cvMatFromUIImage:image];
    
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    cv::Mat cir = cv::Mat::zeros(src.size(), CV_8UC1);
    int bsize = 10;
    
    for (int i = 0; i < src.rows; i += bsize)
    {
        for (int j = 0; j < src.cols; j += bsize)
        {
            cv::Rect rect = cv::Rect(j, i, bsize, bsize) &
            cv::Rect(0, 0, src.cols, src.rows);
            
            cv::Mat sub_dst(dst, rect);
            sub_dst.setTo(cv::mean(src(rect)));
            
            cv::circle(
                       cir,
                       cv::Point(j+bsize/2, i+bsize/2),
                       bsize/2-1,
                       CV_RGB(255,255,255), -1, CV_AA
                       );
        }
    }
    
    cv::Mat cir_32f;
    cir.convertTo(cir_32f, CV_32F);
    cv::normalize(cir_32f, cir_32f, 0, 1, cv::NORM_MINMAX);
    
    cv::Mat dst_32f;
    dst.convertTo(dst_32f, CV_32F);
    
    std::vector<cv::Mat> channels;
    cv::split(dst_32f, channels);
    for (int i = 0; i < channels.size(); ++i)
        channels[i] = channels[i].mul(cir_32f);
    
    cv::merge(channels, dst_32f);
    dst_32f.convertTo(dst, CV_8U);
    
    return [self UIImageFromCVMat:dst];
}

/**
 * Get edges of an image
 * @param gray - grayscale input image
 * @param canny - output edge image
 */
void getCanny(cv::Mat gray, cv::Mat &canny) {
    cv::Mat thres;
    double high_thres = threshold(gray, thres, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU), low_thres = high_thres * 0.5;
    cv::Canny(gray, canny, low_thres, high_thres);
}

struct Line {
    cv::Point _p1;
    cv::Point _p2;
    cv::Point _center;
    
    Line(cv::Point p1, cv::Point p2) {
        _p1 = p1;
        _p2 = p2;
        _center = cv::Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
    }
};

bool cmp_y(const Line &p1, const Line &p2) {
    return p1._center.y < p2._center.y;
}

bool cmp_x(const Line &p1, const Line &p2) {
    return p1._center.x < p2._center.x;
}

/**
 * Compute intersect point of two lines l1 and l2
 * @param l1
 * @param l2
 * @return Intersect Point
 */
cv::Point2f computeIntersect(Line l1, Line l2) {
    int x1 = l1._p1.x, x2 = l1._p2.x, y1 = l1._p1.y, y2 = l1._p2.y;
    int x3 = l2._p1.x, x4 = l2._p2.x, y3 = l2._p1.y, y4 = l2._p2.y;
    if (float d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) {
        cv::Point2f pt;
        pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
        pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
        return pt;
    }
    return cv::Point2f(-1, -1);
}


-(UIImage *)scan:(UIImage* )image isDebug:(Boolean)debug{
    
    /* get input image */
    cv::Mat img = [self cvMatFromUIImage:image];
    
    // resize input image to img_proc to reduce computation
    cv::Mat img_proc;
    int w = img.size().width, h = img.size().height, min_w = 200;
    double scale = fmin(10.0, w * 1.0 / min_w);
    int w_proc = w * 1.0 / scale, h_proc = h * 1.0 / scale;
    resize(img, img_proc, cv::Size(w_proc, h_proc));
    cv::Mat img_dis = img_proc.clone();
    
    /* get four outline edges of the document */
    // get edges of the image
    cv::Mat gray, canny;
    cvtColor(img_proc, gray, CV_BGR2GRAY);
    getCanny(gray, canny);
    
    
    // extract lines from the edge image
    cv::vector<cv::Vec4i> lines;
    cv::vector<Line> horizontals, verticals;
    cv::HoughLinesP(canny, lines, 1, CV_PI / 180, w_proc / 3, w_proc / 3, 20);
    
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i v = lines[i];
        double delta_x = v[0] - v[2], delta_y = v[1] - v[3];
        Line l(cv::Point(v[0], v[1]), cv::Point(v[2], v[3]));
        // get horizontal lines and vertical lines respectively
        if (fabs(delta_x) > fabs(delta_y)) {
            horizontals.push_back(l);
        } else {
            verticals.push_back(l);
        }
        // for visualization only
        if (debug)
            line(img_proc, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), cv::Scalar(0, 0, 255), 1, CV_AA);
    }
    
    // edge cases when not enough lines are detected
    if (horizontals.size() < 2) {
        if (horizontals.size() == 0 || horizontals[0]._center.y > h_proc / 2) {
            horizontals.push_back(Line(cv::Point(0, 0), cv::Point(w_proc - 1, 0)));
        }
        if (horizontals.size() == 0 || horizontals[0]._center.y <= h_proc / 2) {
            horizontals.push_back(Line(cv::Point(0, h_proc - 1), cv::Point(w_proc - 1, h_proc - 1)));
        }
    }
    if (verticals.size() < 2) {
        if (verticals.size() == 0 || verticals[0]._center.x > w_proc / 2) {
            verticals.push_back(Line(cv::Point(0, 0), cv::Point(0, h_proc - 1)));
        }
        if (verticals.size() == 0 || verticals[0]._center.x <= w_proc / 2) {
            verticals.push_back(Line(cv::Point(w_proc - 1, 0), cv::Point(w_proc - 1, h_proc - 1)));
        }
    }
    // sort lines according to their center point
    sort(horizontals.begin(), horizontals.end(), cmp_y);
    sort(verticals.begin(), verticals.end(), cmp_x);
    
    // for visualization only
    if (debug) {
        line(img_proc, horizontals[0]._p1, horizontals[0]._p2, cv::Scalar(0, 255, 0), 2, CV_AA);
        line(img_proc, horizontals[horizontals.size() - 1]._p1, horizontals[horizontals.size() - 1]._p2, cv::Scalar(0, 255, 0), 2, CV_AA);
        line(img_proc, verticals[0]._p1, verticals[0]._p2, cv::Scalar(255, 0, 0), 2, CV_AA);
        line(img_proc, verticals[verticals.size() - 1]._p1, verticals[verticals.size() - 1]._p2, cv::Scalar(255, 0, 0), 2, CV_AA);
    }
    /* perspective transformation */
    
    // define the destination image size: A4 - 200 PPI
    int w_a4 = 1654, h_a4 = 2339;
    //int w_a4 = 595, h_a4 = 842;
    cv::Mat dst = cv::Mat::zeros(h_a4, w_a4, CV_8UC3);
    
    // corners of destination image with the sequence [tl, tr, bl, br]
    cv::vector<cv::Point2f> dst_pts, img_pts;
    dst_pts.push_back(cv::Point(0, 0));
    dst_pts.push_back(cv::Point(w_a4 - 1, 0));
    dst_pts.push_back(cv::Point(0, h_a4 - 1));
    dst_pts.push_back(cv::Point(w_a4 - 1, h_a4 - 1));
    
    // corners of source image with the sequence [tl, tr, bl, br]
    img_pts.push_back(computeIntersect(horizontals[0], verticals[0]));
    img_pts.push_back(computeIntersect(horizontals[0], verticals[verticals.size() - 1]));
    img_pts.push_back(computeIntersect(horizontals[horizontals.size() - 1], verticals[0]));
    img_pts.push_back(computeIntersect(horizontals[horizontals.size() - 1], verticals[verticals.size() - 1]));
    
    // convert to original image scale
    for (size_t i = 0; i < img_pts.size(); i++) {
        // for visualization only
        if (debug) {
            circle(img_proc, img_pts[i], 10, cv::Scalar(255, 255, 0), 3);
        }
        img_pts[i].x *= scale;
        img_pts[i].y *= scale;
    }
    
    // get transformation matrix
    cv::Mat transmtx = getPerspectiveTransform(img_pts, dst_pts);
    
    // apply perspective transformation
    warpPerspective(img, dst, transmtx, dst.size());
    
    //imshow("src", img_dis);
    //imshow("canny", canny);
    //imshow("img_proc", img_proc);
    //imshow("dst", dst);
    
    return [self UIImageFromCVMat:img_proc];
}

cv::Point2f computeIntersect(cv::Vec4i a,
                             cv::Vec4i b)
{
	int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
    
	if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
	{
		cv::Point2f pt;
		pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
		pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
		return pt;
	}
	else
		return cv::Point2f(-1, -1);
}

void sortCorners(std::vector<cv::Point2f>& corners,
                 cv::Point2f center)
{
	std::vector<cv::Point2f> top, bot;
    
	for (int i = 0; i < corners.size(); i++)
	{
		if (corners[i].y < center.y)
			top.push_back(corners[i]);
		else
			bot.push_back(corners[i]);
	}
    
	cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
	cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
	cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
	cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];
    
	corners.clear();
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
	corners.push_back(bl);
}

-(UIImage *)quadwithsegmentation:(UIImage* )image
{
	cv::Point2f center(0,0);
    cv::Mat src = [self cvMatFromUIImage:image];
    
	cv::Mat bw;
	cv::cvtColor(src, bw, CV_BGR2GRAY);
	cv::blur(bw, bw, cv::Size(3, 3));
	cv::Canny(bw, bw, 100, 100, 3);
    
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(bw, lines, 1, CV_PI/180, 70, 30, 10);
    
	// Expand the lines
	for (int i = 0; i < lines.size(); i++)
	{
		cv::Vec4i v = lines[i];
		lines[i][0] = 0;
		lines[i][1] = ((float)v[1] - v[3]) / (v[0] - v[2]) * -v[0] + v[1];
		lines[i][2] = src.cols;
		lines[i][3] = ((float)v[1] - v[3]) / (v[0] - v[2]) * (src.cols - v[2]) + v[3];
	}
    
	std::vector<cv::Point2f> corners;
	for (int i = 0; i < lines.size(); i++)
	{
		for (int j = i+1; j < lines.size(); j++)
		{
			cv::Point2f pt = computeIntersect(lines[i], lines[j]);
			if (pt.x >= 0 && pt.y >= 0)
				corners.push_back(pt);
		}
	}
    
	std::vector<cv::Point2f> approx;
	cv::approxPolyDP(cv::Mat(corners), approx, cv::arcLength(cv::Mat(corners), true) * 0.02, true);
    
	if (approx.size() != 4)
	{
		return nil;
	}
    
	// Get mass center
	for (int i = 0; i < corners.size(); i++)
		center += corners[i];
	center *= (1. / corners.size());
    
	sortCorners(corners, center);
    
	cv::Mat dst = src.clone();
    
	// Draw lines
	for (int i = 0; i < lines.size(); i++)
	{
		cv::Vec4i v = lines[i];
		cv::line(dst, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), CV_RGB(0,255,0));
	}
    
	// Draw corner points
	cv::circle(dst, corners[0], 3, CV_RGB(255,0,0), 2);
	cv::circle(dst, corners[1], 3, CV_RGB(0,255,0), 2);
	cv::circle(dst, corners[2], 3, CV_RGB(0,0,255), 2);
	cv::circle(dst, corners[3], 3, CV_RGB(255,255,255), 2);
    
	// Draw mass center
	cv::circle(dst, center, 3, CV_RGB(255,255,0), 2);
    
	cv::Mat quad = cv::Mat::zeros(300, 220, CV_8UC3);
    
	std::vector<cv::Point2f> quad_pts;
	quad_pts.push_back(cv::Point2f(0, 0));
	quad_pts.push_back(cv::Point2f(quad.cols, 0));
	quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
	quad_pts.push_back(cv::Point2f(0, quad.rows));
    
	cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);
	cv::warpPerspective(src, quad, transmtx, quad.size());
    
	//cv::imshow("image", dst);
	//cv::imshow("quadrilateral", quad);
    
    return [self UIImageFromCVMat:dst];
    
}

-(CGRect)frameForImage:(UIImage*)image inImageViewAspectFit:(UIImageView*)imageView
{
    float imageRatio = image.size.width / image.size.height;
    
    float viewRatio = imageView.frame.size.width / imageView.frame.size.height;
    
    if(imageRatio < viewRatio)
    {
        float scale = imageView.frame.size.height / image.size.height;
        
        float width = scale * image.size.width;
        
        float topLeftX = (imageView.frame.size.width - width) * 0.5;
        
        return CGRectMake(topLeftX, 0, width, imageView.frame.size.height);
    }
    else
    {
        float scale = imageView.frame.size.width / image.size.width;
        
        float height = scale * image.size.height;
        
        float topLeftY = (imageView.frame.size.height - height) * 0.5;
        
        return CGRectMake(0, topLeftY, imageView.frame.size.width, height);
    }
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    if (_imgPickerControll == nil)
    {
        _imgPickerControll = [[UIImagePickerController alloc] init];
        _imgPickerControll.delegate = self;
    }
	// Do any additional setup after loading the view, typically from a nib.
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (IBAction)selectImage:(id)sender {
    //_imgPickerControll.sourceType = UIImagePickerControllerSourceTypeCamera;
    [self presentViewController:_imgPickerControll animated:YES completion:nil];
}

- (IBAction)mosaic:(id)sender {
    _imageView.image = [self CreateMosaicImage:_imageView.image];
}

- (IBAction)canny:(id)sender {
    _imageView.image = [self scan:_imageView.image isDebug:true];
    
    //_imageView.image = [self quadwithsegmentation:_imageView.image];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingImage:(UIImage *)image editingInfo:(NSDictionary *)editingInfo
{
    CGRect rect = CGRectMake(0, 0, image.size.width, image.size.height);
    UIGraphicsBeginImageContext(rect.size);
    [image drawInRect:rect];
    image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    _imageView.image = [image scaleToSize:[self frameForImage:image inImageViewAspectFit:_imageView].size];
    
    [picker dismissModalViewControllerAnimated:NO];
    picker = nil;
}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker
{
    // tell our delegate we are finished with the picker
    //[picker dismissModalViewControllerAnimated:NO];
    [[picker presentingViewController] dismissViewControllerAnimated:NO completion:nil];
    picker = nil;
}

@end
