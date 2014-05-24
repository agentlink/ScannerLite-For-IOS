//
//  ScannerLiteViewController.h
//  ScannerLite
//
//  Created by link on 14-5-24.
//  Copyright (c) 2014年 link. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ScannerLiteViewController : UIViewController<UIImagePickerControllerDelegate, UINavigationControllerDelegate>
@property (strong,nonatomic) UIImagePickerController *imgPickerControll;
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
- (IBAction)selectImage:(id)sender;
- (IBAction)mosaic:(id)sender;
- (IBAction)candy:(id)sender;

@end
