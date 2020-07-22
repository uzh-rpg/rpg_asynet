#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <deque>


namespace py = pybind11;

py::array_t<double> event_queue_tensor(py::array_t<float> events, int queue_length, int H, int W, float time_horizon) 
{
    // find number of events
    auto events_buf = events.request();
    float *events_ptr = (float *) events_buf.ptr;
    int n_events = events_buf.shape[0];

    /*  allocate the buffer */
    py::array_t<float> result = py::array_t<float>({2, queue_length, H, W});
    //py::array_t<int> queue_size = py::array_t<int>({H, W});
    
    auto result_buf = result.request();
    float *result_ptr = (float *) result_buf.ptr;
    
    std::vector<std::deque<std::pair<float,int>>> result_deque(H*W); 

    //auto queue_size_buf = queue_size.request();
    //double *queue_size_ptr = (double *) queue_size_buf.ptr;

    for (size_t idx = 0; idx < n_events; idx++)
    {
        int x = events_ptr[4 * idx + 0];
        int y = events_ptr[4 * idx + 1];
        int p = events_ptr[4 * idx + 3];
        float& t = events_ptr[4 * idx + 2];

        result_deque[x+W*y].push_front({t, p});
        if (result_deque[x+W*y].size()>queue_length)
            result_deque[x+W*y].pop_back();
    }

    for (int i=0; i<H*W; i++)
    {
        auto& e = result_deque[i];
        for (int k=0; k<queue_length; k++)
        {
            if (!e.empty())
            {
                std::pair<float, int> &p = e.front();
                //std::cout << " p.second " << p.second << std::endl;
                result_ptr[i+H*W*k+H*W*queue_length*0] = p.first;
                result_ptr[i+H*W*k+H*W*queue_length*1] = p.second;
                e.pop_front();
            }
            else
            {
                result_ptr[i+H*W*k+H*W*queue_length*0] = 0;
                result_ptr[i+H*W*k+H*W*queue_length*1] = 0;
            }
        }
    }
    
    return result;
}

PYBIND11_MODULE(event_representations, m) {
        m.doc() = "Generate event representations"; // optional module docstring
        m.def("event_queue_tensor", &event_queue_tensor, "Generate event queue tensor");
}