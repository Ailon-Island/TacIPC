#pragma once
#include <zensim/math/Vec.h>
#include <zensim/container/Vector.hpp>
#include <zensim/container/TileVector.hpp>
#include <Eigen/Core>

using json = nlohmann::json;

namespace nlohmann {
template <typename Scalar, int Rows, int Cols>
struct adl_serializer<Eigen::Matrix<Scalar, Rows, Cols>> {
    static void to_json(json& j, const Eigen::Matrix<Scalar, Rows, Cols>& matrix) {
        for (int row = 0; row < matrix.rows(); ++row) {
            nlohmann::json column = json::array();
            for (int col = 0; col < matrix.cols(); ++col) {
                column.push_back(matrix(row, col));
            }
            j.push_back(column);
        }
    }

    static void from_json(const json& j, Eigen::Matrix<Scalar, Rows, Cols>& matrix) {        
        auto setVal = [](Scalar& tgt, const json& value) {
            if constexpr (std::is_same_v<Scalar, bool>)
            {
                if (value.is_string())
                {
                    tgt = value == "true";
                }
                else if (value.is_number())
                {
                    tgt = value != 0;
                }
                else
                {
                    value.get_to(tgt);
                }
            }
            else 
            {
                value.get_to(tgt);
            }
        };
        if (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic) {
            if (j.at(0).is_array())
            {
                matrix.resize(j.size(), j.at(0).size());
            }
            else
            {
                matrix.resize(j.size(), 1);
            }
        }
        for (std::size_t row = 0; row < j.size(); ++row) {
            const auto& jrow = j.at(row);
            if (jrow.is_array())
            {
                for (std::size_t col = 0; col < jrow.size(); ++col) {
                    const auto& value = jrow.at(col);
                    setVal(matrix(row, col), value);
                }
            }
            else 
            {
                setVal(matrix(row, 0), jrow);
            }
        }
    }
}; 

template <typename T, typename AllocatorT>
struct adl_serializer<zs::Vector<T, AllocatorT>> {
    static void to_json(json& j, const zs::Vector<T, AllocatorT>& v) 
    {
        using namespace zs; 

        if (!v.memoryLocation().onHost())
        {
        to_json(j, v.clone({memsrc_e::host, -1}));
        return;
        }

        j["size"] = v.size();
        j["data"] = json::array();
        auto data_ptr = j["data"].get_ptr<json::array_t*>();
        data_ptr->resize(v.size());
        if (v.size() > 0)
        {
            std::copy(v.begin(), v.end(), data_ptr->begin());
        }
    } 
};

template <typename T, auto... Ns>
struct adl_serializer<zs::vec<T, Ns...>> {
    template <auto d, typename VecTM>
    static void _to_json(json& j, const zs::VecInterface<VecTM>& v, auto... inds)
    {
        using namespace zs; 
        constexpr auto dim = VecTM::dim;
        constexpr auto n = VecTM::template range_t<d>::value;

        if constexpr (d == dim - 1) {
            auto vec_ptr = j.get_ptr<json::array_t*>();
            vec_ptr->resize(n);
            for (int i = 0; i < n; ++i) {
                (*vec_ptr)[i] = v(inds..., i);
            }
        } 
        else 
        {
            for (int i = 0; i < n; ++i) {
                j[i] = std::vector{n, json::array_t{}};
                _to_json<d + 1>(j[i], v, inds..., i);
            }
        }
    }

    static void to_json(json& j, const zs::vec<T, Ns...>& v) 
    {
        using namespace zs; 
        constexpr auto dim = sizeof...(Ns);

        // j["shape"] = std::vector<int>{Ns...};
        j = json::array_t();
        _to_json<0>(j, v);
    } 
};

template <>
struct adl_serializer<zs::PropertyTag>
{
    static void to_json(json& j, const zs::PropertyTag& pt)
    {
        j = json{ {"name", pt.name.asChars()}, {"numChannels", pt.numChannels} };
    }
};


template <typename T, size_t Length, typename AllocatorT>
struct adl_serializer<zs::TileVector<T, Length, AllocatorT>> {
    static void to_json(json &j, const zs::TileVector<T, Length, AllocatorT> &tv)
    {
        using namespace zs;

        auto size = tv.size();

        if (!tv.memoryLocation().onHost()) 
        {
            to_json(j, tv.clone({memsrc_e::host, -1}));
            return;
        }

        auto pol = omp_exec();
        constexpr auto space = execspace_e::openmp;
        auto vw = view<space>({}, tv, false_c, "to json");

        const auto &propertyTags = tv.getPropertyTags();
        const auto &propertyTagNames = vw.getPropertyNames();
        const auto &propertyOffs = vw.getPropertyOffsets();
        const auto &propertySizes = vw.getPropertySizes();
        j["propertyTags"] = json::array();
        j["propertyTags"].get_ptr<json::array_t*>()->resize(propertyTags.size());
        // j["propertyTags"] = tv.getPropertyTags();

        for (int i = 0; i < propertyTags.size(); i++)
        {
            j["propertyTags"][i] = propertyTags[i];
            auto &tagName = propertyTagNames[i];
            auto &offset = propertyOffs[i];
            auto &psize = propertySizes[i];

            j["data"][tagName.asChars()] = json::array();

            if (size == 0)
                continue;
            
            auto ppt = json::array();
            auto ppt_ptr = ppt.get_ptr<json::array_t*>();
            ppt_ptr->resize(size);
            if (psize == 1)
            {
                pol(range(size),
                    [&](int index) mutable
                    {
                        (*ppt_ptr)[index] = vw(offset, index);
                    });
            }
            else 
            {
                pol(range(size),
                    [&](int index) mutable
                    {
                        (*ppt_ptr)[index] = json::array();
                        auto ptr = (*ppt_ptr)[index].get_ptr<json::array_t*>();
                        ptr->resize(psize);
                        for (int d = 0; d < psize; d++)
                            (*ptr)[d] = vw(offset + d, index);
                    });
            }
            j["data"][tagName.asChars()] = std::move(ppt);
        }
    }
};

} // namespace nlohmann