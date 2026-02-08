MYDETAPI retdata GetImagePtr_Process(
    retHandle handle,
    unsigned char* image_data,   // 图像数据指针
    int32_t width,               // 图像宽度
    int32_t height,              // 图像高度
    int32_t channels             // 图像通道数
) {
    retdata result={0};
    if (!handle.ProcessorDetectPtr) {
        std::cerr << "DetectImage error: Invalid handle" << std::endl;
        result.anomal_type = -2;
        return result;  // 错误码
    }
    try {
        int max_detections = 20;
        // 为 detections 分配内存
        Detection* detections = (Detection*)malloc(max_detections * sizeof(Detection));
        if (detections == NULL) {
            // 处理内存分配失败的情况
            fprintf(stderr, "Failed to allocate memory for detections\n");
            result.anomal_type = -3;
            return result;  // 错误码
        }
        int ret = DetectImage(handle.ProcessorDetectPtr, image_data, width, height, channels, detections, max_detections);
        std::cout << "detect  " << ret << "  box" << std::endl;
        if (ret > 0) {
            // 成功检测到 result 个目标
            cv::Mat image(height, width, channels, image_data);
            for (int i = 0; i < ret; ++i) {
                std::cout << "Detection: Class=" << detections[i].class_id << ", Confidence=" << detections[i].confidence
                    << ", x=" << detections[i].x << ", y=" << detections[i].y
                    << ", width=" << detections[i].width << ", height=" << detections[i].height << std::endl;
                cv::Rect rect(detections[i].x, detections[i].y, detections[i].width, detections[i].height);
                cv::Mat croped = image(rect);
                unsigned char* data = croped.data;
                int width = croped.cols;
                int height = croped.rows;
                int channels = croped.channels();
                if (detections[i].class_id == 0 and handle.ProcessorAnomalPtr1) {
                    int ret = SetImage(handle.ProcessorAnomalPtr1, data, width, height, channels);
                    if (ret == 1)
                    {
                        anomalret res = GetResultsMap(handle.ProcessorAnomalPtr1);
                        if (res.type == 1) {
                            std::cout << ": pass!" << std::endl;
                        }
                        else if (res.type == -1) {
                            cv::Mat resultRegion(res.height,res.width, res.channels, res.image_data);
                            resultRegion.copyTo(image(rect));
                            result.anomal_type = -1;
                           
                        }
                    }
                    else
                    {
                        printf("setimage to pcb_1 error occurred\n");
                    }
                }
                if (detections[i].class_id == 1 and handle.ProcessorAnomalPtr2) {
                    int ret = SetImage(handle.ProcessorAnomalPtr2, data, width, height, channels);
                    if (ret == 1)
                    {
                        anomalret res = GetResultsMap(handle.ProcessorAnomalPtr2);
                        if (res.type == 1) {
                            std::cout << ": pass!" << std::endl;
                        }
                        else if (res.type == -1) {
                            cv::Mat resultRegion(res.height, res.width, res.channels, res.image_data);
                            resultRegion.copyTo(image(rect));
                            result.anomal_type = -1;
                        }
                    }
                    else
                    {
                        printf("setimage to pcb_2 error occurred\n");
                    }
                }
                if (detections[i].class_id == 2 and handle.ProcessorAnomalPtr3) {
                    int ret = SetImage(handle.ProcessorAnomalPtr3, data, width, height, channels);
                    if (ret == 1)
                    {
                        anomalret res = GetResultsMap(handle.ProcessorAnomalPtr1);
                        if (res.type == 1) {
                            std::cout << ": pass!" << std::endl;
                        }
                        else if (res.type == -1) {
                            cv::Mat resultRegion(res.height, res.width, res.channels, res.image_data);
                            resultRegion.copyTo(image(rect));
                            result.anomal_type = -1;
                        }
                    }
                    else
                    {
                        printf("setimage to pcb_3 error occurred\n");
                    }
                }
                if (detections[i].class_id == 3 and handle.ProcessorAnomalPtr4) {
                    int ret = SetImage(handle.ProcessorAnomalPtr4, data, width, height, channels);
                    if (ret == 1)
                    {
                        anomalret res = GetResultsMap(handle.ProcessorAnomalPtr4);
                        if (res.type == 1) {
                            std::cout << ": pass!" << std::endl;
                        }
                        else if (res.type == -1) {
                            cv::Mat resultRegion(res.height, res.width, res.channels, res.image_data);
                            resultRegion.copyTo(image(rect));
                            result.anomal_type = -1;
                        }
                    }
                    else
                    {
                        printf("setimage to pcb_4 error occurred\n");
                    }
                }
                if (detections[i].class_id == 4 and handle.ProcessorAnomalPtr5) {
                    int ret = SetImage(handle.ProcessorAnomalPtr5, data, width, height, channels);
                    if (ret == 1)
                    {
                        anomalret res = GetResultsMap(handle.ProcessorAnomalPtr5);
                        if (res.type == 1) {
                            std::cout << ": pass!" << std::endl;
                        }
                        else if (res.type == -1) {
                            cv::Mat resultRegion(res.height, res.width, res.channels, res.image_data);
                            resultRegion.copyTo(image(rect));
                            result.anomal_type = -1;
                        }
                    }
                    else
                    {
                        printf("setimage to pcb_1 error occurred\n");
                    }
                }
                if (detections[i].class_id == 5 and handle.ProcessorAnomalPtr6) {
                    int ret = SetImage(handle.ProcessorAnomalPtr6, data, width, height, channels);
                    if (ret == 1)
                    {
                        anomalret res = GetResultsMap(handle.ProcessorAnomalPtr6);
                        if (res.type == 1) {
                            std::cout << ": pass!" << std::endl;
                        }
                        else if (res.type == -1) {
                            cv::Mat resultRegion(res.height, res.width, res.channels, res.image_data);
                            resultRegion.copyTo(image(rect));
                            result.anomal_type = -1;
                        }
                    }
                    else
                    {
                        printf("setimage to pcb_6 error occurred\n");
                    }
                }
                if (detections[i].class_id == 6 and handle.ProcessorAnomalPtr7) {
                    int ret = SetImage(handle.ProcessorAnomalPtr7, data, width, height, channels);
                    if (ret == 1)
                    {
                        anomalret res = GetResultsMap(handle.ProcessorAnomalPtr7);
                        if (res.type == 1) {
                            std::cout << ": pass!" << std::endl;
                        }
                        else if (res.type == -1) {
                            cv::Mat resultRegion(res.height, res.width, res.channels, res.image_data);
                            resultRegion.copyTo(image(rect));
                            result.anomal_type = -1;
                        }
                    }
                    else
                    {
                        printf("setimage to pcb_7 error occurred\n");
                    }
                }
                if (detections[i].class_id == 7 and handle.ProcessorAnomalPtr8) {
                    int ret = SetImage(handle.ProcessorAnomalPtr8, data, width, height, channels);
                    if (ret == 1)
                    {
                        anomalret res = GetResultsMap(handle.ProcessorAnomalPtr8);
                        if (res.type == 1) {
                            std::cout << ": pass!" << std::endl;
                        }
                        else if (res.type == -1) {
                            cv::Mat resultRegion(res.height, res.width, res.channels, res.image_data);
                            resultRegion.copyTo(image(rect));
                            result.anomal_type = -1;
                        }
                    }
                    else
                    {
                        printf("setimage to pcb_1 error occurred\n");
                    }
                }
                if (detections[i].class_id == 8 and handle.ProcessorAnomalPtr9) {
                    int ret = SetImage(handle.ProcessorAnomalPtr9, data, width, height, channels);
                    if (ret == 1)
                    {
                        anomalret res = GetResultsMap(handle.ProcessorAnomalPtr9);
                        if (res.type == 1) {
                            std::cout << ": pass!" << std::endl;
                        }
                        else if (res.type == -1) {
                            cv::Mat resultRegion(res.height, res.width, res.channels, res.image_data);
                            resultRegion.copyTo(image(rect));
                            result.anomal_type = -1;
                        }
                    }
                    else
                    {
                        printf("setimage to pcb_1 error occurred\n");
                    }
                }
            }
        }
        else {
            // 没有检测到目标或者出错
            printf("No detections or error occurred\n");
        }
        std::cout << "process over" << std::endl;
        free(detections);
        result.image_data = image_data;
        result.width = width;
        result.height = height;
        result.channels = channels;
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "box detection  error: " << e.what() << std::endl;
        result.anomal_type = -2;
        return result;  // 错误码
    }
    catch (...) {
        std::cerr << "Unknown error in CreateProcessor" << std::endl;
        result.anomal_type = -4;
        return result;  // 错误码
    }

}
