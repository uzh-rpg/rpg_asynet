#pragma once
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include <rulebook.h>


template<class MatrixType>
struct ResizableRef
{
    typedef typename MatrixType::Scalar Scalar;
    class MatrixDummy;
    typedef void (MatrixDummy::*ResizeFunctor)(Eigen::Index rows, Eigen::Index Cols);
    typedef Scalar* (MatrixDummy::*DataGetter)();

    MatrixDummy *m;
    const ResizeFunctor resizer;
    const DataGetter getData;

    template<class Derived>
    ResizableRef(Eigen::MatrixBase<Derived>& M)
      : m(reinterpret_cast<MatrixDummy*>(&M))
      , resizer(reinterpret_cast<ResizeFunctor>((void (Derived::*)(Eigen::Index, Eigen::Index)) &Derived::resize))
      , getData(reinterpret_cast<DataGetter>((Scalar* (Derived::*)()) &Derived::data))
    { }

    template<class Derived>
    ResizableRef& operator=(const Eigen::EigenBase<Derived>& other)
    {
        (m->*resizer)(other.rows(), other.cols());
        MatrixType::Map((m->*getData)(), other.rows(), other.cols()) = other;
    }
};

class AsynSparseConvolution2D
{
public:
    AsynSparseConvolution2D(int dimension, int nIn, int nOut, int filter_size, bool first_layer, bool use_bias, bool debug);
    ~AsynSparseConvolution2D();

    enum Site
    {
        INACTIVE,
        ACTIVE,
        UPDATED,
        NEW_ACTIVE,
        NEW_INACTIVE,
        VISUALIZATION_UPDATE
    };

    using ActiveMatrix = Eigen::Matrix<Site, -1, 1>;
    using ReturnType = std::tuple<Eigen::MatrixXi, Eigen::MatrixXf, ActiveMatrix>;
    ReturnType forward(const Eigen::Ref<const Eigen::MatrixXi> update_location,
                       const Eigen::Ref<const Eigen::MatrixXf> input_feature_map, 
                       Eigen::Ref<ActiveMatrix>& active_sites_map,
                       RuleBook& rule_book,
                       bool no_update_locations);
    
    void setParameters(Eigen::VectorXf bias, Eigen::MatrixXf weights);

    void initMaps(int H, int W);
    ActiveMatrix initActiveMap(Eigen::MatrixXf input_feature_map, const Eigen::MatrixXi update_location);

private:
    void updateRulebooks(Eigen::Matrix<bool,-1,1> bool_new_active_site,
                         Eigen::Matrix<bool,-1,1> zero_input_update,
                         Eigen::VectorXi update_location_linear,
                         Eigen::Ref<ActiveMatrix>& active_sites_map, 
                         Eigen::VectorXi& new_update_location_linear,
                         RuleBook& rulebook);

    int dimension_;
    int nIn_;
    int nOut_;
    int filter_size_;
    int filter_volume_;
    int num_pixels_;
    int H_;
    int W_;
    
    bool first_layer_;
    bool use_bias_;
    bool initialized_output_maps_;
    bool initialized_input_maps_;
    bool debug_;

    Eigen::VectorXi filter_size_tensor_;
    
    Eigen::MatrixXf weights_;
    Eigen::VectorXf bias_;

    Eigen::VectorXi padding_;

    Eigen::MatrixXi kernel_indices_;
    
    Eigen::MatrixXf output_feature_map_;
    Eigen::MatrixXf old_input_feature_map_;

};