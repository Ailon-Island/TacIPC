#include <tacipc/constitutiveModel/constitutiveModel.cuh>

namespace tacipc
{
template struct StVKWithHenckyModel<float>;
template struct StVKWithHenckyModel<double>;

template struct NeoHookeanModel<float>;
template struct NeoHookeanModel<double>;

template struct BaraffWitkinModel<float>;
template struct BaraffWitkinModel<double>;

template struct OrthogonalModel<float>;
template struct OrthogonalModel<double>;

template struct ConstitutiveModel<float, BodyType::Soft>;
template struct ConstitutiveModel<double, BodyType::Soft>;
template struct ConstitutiveModel<float, BodyType::Rigid>;
template struct ConstitutiveModel<double, BodyType::Rigid>;
} // namespace tacipc