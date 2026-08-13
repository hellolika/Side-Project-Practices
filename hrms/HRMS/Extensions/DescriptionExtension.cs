using System.ComponentModel;

namespace HRMS.Extensions
{
    public static class DescriptionExtension
    {
        public static string GetDescription<T>(this T enumParameter)
        {
            var type = enumParameter.GetType();
            if (!type.IsEnum) throw new ArgumentException("EnumParameter is not the type of Enum", nameof(enumParameter));

            var memberInfo = type.GetMember(enumParameter.ToString());
            if (memberInfo.Length <= 0) return enumParameter.ToString();

            var attrs = memberInfo[0].GetCustomAttributes(typeof(DescriptionAttribute), false);
            return attrs.Length > 0 ? ((DescriptionAttribute)attrs[0]).Description : enumParameter.ToString();
        }
    }
}
