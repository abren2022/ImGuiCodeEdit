		vector<cv::Mat> images = gen_images(image, anomaly_map, score, this->meta.pixel_threshold, image_type);
		cv::Mat res = images[0];
		size_t dataSize = res.total() * res.elemSize();
		ret.image_data = new unsigned char[dataSize];
		memcpy((void*)ret.image_data, res.data, dataSize);
		//ret.image_data = res.data;
		ret.width = res.cols;
		ret.height = res.rows;
		ret.channels = res.channels();
		ret.type = image_type;

		return ret;
	}

    int ret = SetImage(handle.ProcessorAnomalPtr1, croped.data, croped.cols, croped.rows, croped.channels());
                    if (ret == 1)
                    {
                        anomalret res1 = GetResultsMap(handle.ProcessorAnomalPtr1);
                        std::cout << "1 type: " << res1.type << std::endl;
                        if (res1.type == 1) {
                            std::cout << ": pass!" << std::endl;
                        }
                        else if (res1.type == -1) {
                            cv::Mat resultRegion(res1.height,res1.width, CV_8UC3, res1.image_data);
                            cv::imshow("显示窗口", resultRegion);
                            cv::waitKey(0);
                            //resultRegion.copyTo(image(rect));
                            result.anomal_type = -1;
                            resultRegion.release();
                           
                        }
                        //delete[] res1.image_data;
                        //res1.image_data = nullptr;
                    }
                    else
                    {
                        printf("setimage to pcb_1 error occurred\n");
                    }
