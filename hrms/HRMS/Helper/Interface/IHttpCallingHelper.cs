using HRMS.Enum;
using HRMS.Models.Requests;

namespace HRMS.Helper;

public interface IHttpCallingHelper
{
    Task<T1> GetCalling<T1>(string address, Dictionary<string, string> headers = null) where T1 : class;
    Task<T1> PostCalling<T1,T2>(string apiKey, string address,T2 request,EnumHttpContentType enumHttpContentType) where T1 : class where T2:class;
    
    Task<T1> SendNotification<T1, T2>(string address, T2 request, EnumHttpContentType enumHttpContentType,
        string apiKey)  where T2 : OneSignalNotificationRequestBase;
}