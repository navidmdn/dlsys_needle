#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;

using namespace std;


void apply_relu(vector<vector<double> > *vec) {
    for (auto & row : *vec) {
        for (auto & x: row)
            x = std::max(x, 0.0);
    }
}

void apply_exp(vector<vector<double> > *vec) {
    for (auto & row : *vec) {
        for (auto & x : row)
            x = exp(x);
    }
}

void scale(vector<vector<double> > *vec, double num) {
    for (auto & row : *vec) {
        for (auto & x : row)
            x *= num;
    }
}

float** matmul2d(float** m1, float** m2, int m1_1, int m1_2, int m2_2) {
    float** res = new float*[m1_1];
    for (int i = 0; i < m1_1; ++i)
        res[i] = new float[m2_2]();
        
        
    for (int i = 0; i < m1_1; i++) {
        for (int j = 0; j < m2_2; j++) {
            for (int k = 0; k < m1_2; k++)
                res[i][j] += m1[i][k]*m2[k][j];
        }
    }
    
    return res;
}

vector<vector<double> > matmul2d(vector<vector<double> > m1, vector<vector<double> > m2) {
    vector<vector<double> > res(m1.size(), vector<double> (m2[0].size(), 0));

    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size(); j++) {
            for (int k = 0; k < m2.size(); k++)
                res[i][j] += m1[i][k]*m2[k][j];
        }
    }

    return res;
}


vector<vector<double> > reshape(const float *inp, int beg, int n, int m) {
    vector<vector<double> > res(n, vector<double> (m, 0));
    
    for (int i = beg; i < beg + n*m; i++) {
        auto idx = i - beg;
        res[idx/m][idx%m] = inp[i];
    }
    
    return res;
}

vector<vector<double> > slice(vector<vector<double> > vec, int beg, int end) {
    vector<vector<double> > res;
    
    for (int i = beg; i < end; i++) {
        auto v = vec[i];
        res.push_back(v);
    }
    
    return res;
}


vector<vector<double> > transpose(vector<vector<double> > vec) {
    auto nrow = vec.size();
    auto ncol = vec[0].size();

    vector<vector<double> > res;

    for (int j = 0; j < ncol; j++){
        vector<double> cur_row;
        for (int i = 0; i < nrow; i++) 
            cur_row.push_back(vec[i][j]);
        res.push_back(cur_row);
    }

    return res;
}

vector<vector<double> > row_sum(vector<vector<double> > vec) {
    vector<vector<double> > res;

    for (auto row : vec) {
        auto _sum = 0.0;
        vector<double> sum_row;
        for (auto x : row) {
            _sum += x;
        }
        sum_row.push_back(_sum);
        res.push_back(sum_row);
    }

    return res;
}

vector<vector<double> > add_mat(vector<vector<double> > vec1, vector<vector<double> > vec2) {
    vector<vector<double> > res;

    //todo: check size mismatch
    for (int i = 0; i < vec1.size(); i++) {
        vector<double> row;
        for (int j = 0; j < vec1[0].size(); j++) {
            row.push_back(vec1[i][j]+vec2[i][j]);
        }
        res.push_back(row);
    }
    return res;
}


vector<vector<double> > one_hot(const unsigned char *y, int size, int n_targets) {
    vector<vector<double> > res(size, vector<double> (n_targets, 0.0));

    for (int i = 0; i < size; i++) {
        res[i][y[i]] = 1.0;
    }

    return res;
}

void assign(float* mat2d, vector<vector<double> > vals) {
    for (int i = 0; i < vals.size(); i++) {
        for (int j = 0; j < vals[0].size(); j++) {
            mat2d[i*vals[0].size()+j] = vals[i][j];
        }
    }
}

void print_mat(vector<vector<double> > mat) {
    cout << "_____________" << endl;
    for (auto row : mat) {
        for (auto x : row)
            cout << x << " ";
        cout << endl;
    }
    cout << "_____________" << endl;
}

void get_shape(vector<vector<double> > x) {
    cout << x.size() << " x " << x[0].size() << endl;
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int i = 0;
    vector<vector<double> > theta2d = reshape(theta, 0, n, k);;
    cout << "initial theta:" << endl;
    print_mat(theta2d);
    
    auto onehot_y = one_hot(y, m, k);
    
    
    while (i < m) {
        
        // todo: handle last batch's mismatch size

        //get the current batch
        int begin_x_idx = i*n;
        cout << "begin x idx:" << begin_x_idx << endl;

        int cur_batch_size = min(batch, m-i);
        auto cur_x = reshape(X, begin_x_idx, cur_batch_size, n);
        auto cur_y = slice(onehot_y, i, i+cur_batch_size);
        cout << "batch size:" << endl;
        get_shape(cur_x);
    
        cout << "theta:" << endl;
        get_shape(theta2d);
        // print_mat(cur_x);

        auto z = matmul2d(cur_x, theta2d);
        
        cout << "X x th=" <<endl;
        print_mat(z);
        
        cout << "z:" << endl;
        get_shape(z);

        apply_exp(&z);
        
        
        cout << "exp()=" <<endl;
        print_mat(z);
        
        cout << "z:" << endl;
        get_shape(z);

        auto row_sum_z = row_sum(z);
        cout << "row sum:" << endl;
        get_shape(row_sum_z);
        
        
        
        // divide by row sum (broadcasting)
        for (auto i = 0; i < row_sum_z.size(); i++) {
            double sum = row_sum_z[i][0];
            for (int j = 0; j < z[0].size(); j++) {
                z[i][j] /= sum;
            }
        }
        
        
        cout << "normal z=" <<endl;
        print_mat(z);
        

        cout << "z:" << endl;
        get_shape(z);

        cout << "cur_y" << endl;
        get_shape(cur_y);
        
        cout << "current y" << endl;
        print_mat(cur_y);
        
        scale(&cur_y, -1.0);
        auto grad = matmul2d(transpose(cur_x), add_mat(z, cur_y));
        scale(&grad, 1.0/cur_batch_size);

        cout << "grad:" << endl;
        get_shape(grad);

        scale(&grad, double(-1.0 * lr));
        theta2d = add_mat(theta2d, grad);

        cout << "final updated theta:" << endl;
        print_mat(theta2d);

        i += batch;
    }
    assign(theta, theta2d);
            
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}

// int main(int argc, char const *argv[])
// {
//     // vector<vector<double>> a{{1,2,3},{2,3,4},{0,1,1}};
//     // vector<vector<double>> b{{1}, {0}, {2}};
    
//     // print_mat(a);
//     // print_mat(b);

//     const float X[9] = {1,2,3,2,-3,4,0,1,1};
//     const float *Xpt = X;
//     const unsigned char y[3] = {1, 2, 3};
//     const unsigned char *ypt = y;
//     float *th = new float(10);
    
//     //const float* y = X;

//     softmax_regression_epoch_cpp(Xpt, ypt, th, 3, 3, 10, 0.1, 2);
//     //auto x = matmul2d(a, b);

//     //print_mat(x);

//     return 0;
// }
