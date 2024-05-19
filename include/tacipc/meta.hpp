#pragma once
#include <tacipc/types.hpp>

namespace tacipc
{
// https://stackoverflow.com/questions/66363972/how-to-cause-static-error-in-constexpr-if-else-chain
template <int i> struct dependent_false : std::false_type
{
};

// concat two variant types
template <class T1, class T2>
struct variant_cat{};
template <class... t1s, class... t2s>
struct variant_cat<std::variant<t1s...>, std::variant<t2s...>>
{
    using type = std::variant<t1s..., t2s...>;
};

// wrap types in variant with templates
template <class T, template <typename> class... Ts>
struct variant_wrap{};
template <template <typename> class... Ts, typename t, template <typename...> class T>
struct variant_wrap<T<t>, Ts...>
{
    using type = std::variant<Ts<t>...>;
};
template <template <typename> class... Ts, 
    typename t0, typename... ts,
    template <typename...> class T
>
struct variant_wrap<T<t0, ts...>, Ts...>
{
    using type = typename variant_cat<
            std::variant<Ts<t0>...>, 
            typename variant_wrap<T<ts...>, Ts...>::type
        >::type;
};
// wrap the floating point type variant with templates
template <template <typename> class... Ts>
struct variant_wrap_floating_point
{
    using type = typename variant_wrap<floating_point_t, Ts...>::type;
};

// check if a type is a member of a variant
template <typename t, class T>
constexpr bool is_variant_member_v = false;
template <typename t, class... ts>
constexpr bool is_variant_member_v<t, std::variant<ts...>> = std::disjunction<std::is_same<t, ts>...>::value;


/// property map
/// key: a char sequence for property name
/// value: an size_t sequence for property shape
/// usage:
///     using dict = decltype(prop_map(
///         prop_item(string, index_sequence),
///         prop_item(string, size0, size1, size2, ...),
///         prop_item(char_sequence, index_sequence),
///         prop_item(char_sequence, size0, size1, size2, ...),
///         ...
///     ));
///     using value = dict::get_t<some_string>;
///     constexpr int index = dict::get_index<some_string>;
template <char... cs>
using char_sequence = std::integer_sequence<char, cs...>;

// convert const string to char sequence
template <const char* str, class T>
struct str_to_char_seq_helper;
template <const char* str, std::size_t... inds>
struct str_to_char_seq_helper<str, std::index_sequence<inds...>> {
private:
    constexpr static auto str_view = std::string_view{str};
public:
    using type = std::integer_sequence<char, str_view[inds]...>;
};
template <const char* str>
struct str_to_char_seq {
private:
    constexpr static auto str_view = std::string_view{str};
    using inds = std::make_index_sequence<str_view.size()>;
public:
    using type = typename str_to_char_seq_helper<str, inds>::type;
};
template <const char* str>
using str_to_char_seq_t = typename str_to_char_seq<str>::type;


// compile-time property map
template <class Key, class Value>
struct propItem {
    using key_type = Key;
    using value_type = Value;
};
template <char... chars, std::size_t... sizes> 
struct propItem<char_sequence<chars...>, std::index_sequence<sizes...>> {
    using key_type = char_sequence<chars...>;
    using value_type = std::index_sequence<sizes...>;
};
template <int idx = 0, class...>
struct propMap_; 
template <int idx>
struct propMap_<idx> {
    template <class Q> struct get_ { 
        using type = void; 
        constexpr static int index = -1; 
    }; 
}; 
template <int idx, class Item, class... Rest>
struct propMap_<idx, Item, Rest...> {
    template <class Query>
    struct get_ {
        using type = std::conditional_t<std::is_same_v<typename Item::key_type, Query>, 
            typename Item::value_type, typename propMap_<idx+1, Rest...>::template get_<Query>::type>; 
        constexpr static int index = (std::is_same_v<typename Item::key_type, Query>)?idx:propMap_<idx+1, Rest...>::template get_<Query>::index;
    }; 
    template <const char* Qstr>
    struct get{
        using type = typename get_<typename str_to_char_seq<Qstr>::type>::type;
        constexpr static int index = get_<typename str_to_char_seq<Qstr>::type>::index;
    };
    template <const char* Qstr>
    using get_t = typename get<Qstr>::type;
    template <const char* Qstr>
    constexpr static int get_index = get<Qstr>::index;
}; 
template <class... Items>
struct propMap : propMap_<0, Items...> {};
template <class Key, class Value>
constexpr auto prop_item()
{
    return propItem<Key, Value>{};
}
template <const char* str, class Value>
constexpr auto prop_item()
{
    return propItem<str_to_char_seq_t<str>, Value>{};
}
template <class Key, std::size_t... vs>
constexpr auto prop_item()
{
    return propItem<Key, std::index_sequence<vs...>>{};
}
template <const char* str, std::size_t... vs>
constexpr auto prop_item()
{
    return propItem<str_to_char_seq_t<str>, std::index_sequence<vs...>>{};
}
template<class... Args>
constexpr auto prop_map(Args... args)
{
    return propMap<decltype(args)...>{};
}


} // namespace tacipc