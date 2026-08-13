using System.Text;
using HRMS.Enum;
using HRMS.Models.Requests;
using HRMS.Services.Interfaces;
using Newtonsoft.Json;

namespace HRMS.Helper;

public class HttpCallingHelper : IHttpCallingHelper
{
      private readonly HttpClient _client;
        private readonly ILoggerService _logger;

        public HttpCallingHelper(ILoggerService logger, HttpClient httpClient)
        {
            _logger = logger;
            _client = httpClient;
        }

        public async Task<T1> GetCalling<T1>(string address, Dictionary<string, string> headers = null) where T1 : class
        {
            try
            {
                if (headers != null)
                {
                    foreach (var header in headers)
                    {
                        _client.DefaultRequestHeaders.Remove(header.Key);
                        _client.DefaultRequestHeaders.Add(header.Key, header.Value);
                    }
                }

                _logger.Info($"[][Get] Request to url : {address}");
                var result = _client.GetAsync(address).Result;
                _logger.Error(
                    $"Test Error ============= {JsonConvert.SerializeObject(result)} ");
                if (result.IsSuccessStatusCode)
                {
                    var responseBody = await result.Content.ReadAsStringAsync();
                    _logger.Info($"[][Get] Response to url : {responseBody}");
                    return JsonConvert.DeserializeObject<T1>(responseBody);
                }
 
                _logger.Error(
                    $"[][Get]  Response With ErrorCode url : {address} : ErrorCode : {result.StatusCode} , Stack : {result.ReasonPhrase}");
            }
            catch (Exception e)
            {
                _logger.Error(
                    $"[][Get]  Response With ErrorCode url : {address} : ErrorCode : {e.Message} , Stack : {e.StackTrace}");
            }
            return default;
        }
        
        
        public async Task<T1> PostCalling<T1, T2>(string apiKey, string address, T2 request, EnumHttpContentType enumHttpContentType) where T1 : class where T2: class
        {
            // var hashCode = RandomHelper.RandomString(EnumRandomStringType.CharSet, 8);
            try
            {
                if (apiKey != null)
                {
                    _client.DefaultRequestHeaders.Remove("Authorization");
                    _client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
                }

                _logger.Info(
                    $"[[Post] Request to url : {address} , request : {JsonConvert.SerializeObject(request)} ");
                var data = JsonConvert.SerializeObject(request);
                var jsonRequestContent = new StringContent(data, Encoding.UTF8, "application/json");
                _client.DefaultRequestHeaders.ConnectionClose = true;
                var result = await _client.PostAsync(address, jsonRequestContent);
                if (result.IsSuccessStatusCode)
                {
                    var responseBody = await result.Content.ReadAsStringAsync();
                    if (!address.Contains("get-lobby-info.aspx"))
                        _logger.Info($"[][Post]  Response url : {address} , responseBody : {responseBody}");

                    // return JsonConvert.DeserializeObject<T1>(responseBody);
                }
                
                return default;
            }
            catch (Exception e)
            {
                _logger.Error(
                    $"[][Post]  Response With ErrorCode url : {address} : ErrorCode : {e.Message} , Stack : {e.StackTrace}");
            }

            return default;
        }
        
        

        public async Task<T1> SendNotification<T1, T2>(string address, T2 request,
            EnumHttpContentType enumHttpContentType, string apiKey) where T2 : OneSignalNotificationRequestBase
        {
            // var hashCode = RandomHelper.RandomString(EnumRandomStringType.CharSet, 8);
            try
            {
                if (apiKey != null)
                {
                    _client.DefaultRequestHeaders.Remove("Authorization");
                    _client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
                }

                _logger.Info(
                    $"[[Post] Request to url : {address} , request : {JsonConvert.SerializeObject(request)} ");
                var data = JsonConvert.SerializeObject(request);
                var jsonRequestContent = new StringContent(data, Encoding.UTF8, "application/json");
                _client.DefaultRequestHeaders.ConnectionClose = true;
                var result = await _client.PostAsync(address, jsonRequestContent);
                if (result.IsSuccessStatusCode)
                {
                    var responseBody = await result.Content.ReadAsStringAsync();
                    if (!address.Contains("get-lobby-info.aspx"))
                        _logger.Info($"[][Post]  Response url : {address} , responseBody : {responseBody}");

                    return JsonConvert.DeserializeObject<T1>(responseBody);
                }

                _logger.Error(
                    $"[][Post]  Response With Not SuccessStatusCode : {address} : StatusCode: {result.StatusCode} Body: {await result.Content.ReadAsStringAsync()}");

                return default;
            }
            catch (Exception e)
            {
                _logger.Error(
                    $"[][Post]  Response With ErrorCode url : {address} : ErrorCode : {e.Message} , Stack : {e.StackTrace}");
            }

            return default;
        }
}