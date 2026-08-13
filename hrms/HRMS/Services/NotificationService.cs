using System.Text;
using HRMS.Enum;
using HRMS.Helper;
using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Services.Interfaces;

namespace HRMS.Services;

public class NotificationService : INotificationService
{
    private readonly IHttpCallingHelper _http;

    public NotificationService(IHttpCallingHelper http)
    {
        _http = http;
    }
    
    public async Task<ApiBaseResponse<SendNotificationResponse>> SendNotificationByExternalIds(SendNotificationByExternalIdRequest request)
    {
        var sendNotificationResponse = new SendNotificationResponse();
        var headings = request.Messages.ToDictionary(m => m.Language, m => m.Title);
        var contents = request.Messages.ToDictionary(m => m.Language, m => m.Content);
        var oneSignalRequest = new OneSignalNotificationRequestBase()
        {
            Headings = headings,
            Contents = contents,
            AppId = "d4ea2d90-0856-4162-afc4-f488815a9357",
            ExternalIds = request.ExternalIds,
        };
        
        var apiBaseResponse = await OneSignalPushNotification(oneSignalRequest);

        if (apiBaseResponse.ErrorCode == ApiErrorEnum.NoError)
        {
            var data = apiBaseResponse.Result;
            sendNotificationResponse.Id = data.Id;
            sendNotificationResponse.Recipients = data.Recipients;
            sendNotificationResponse.SentAt = data.SentAt;
            
        }
        return new ApiBaseResponse<SendNotificationResponse>(sendNotificationResponse);
    }

    public async Task<ApiBaseResponse<SendNotificationResponse>> SendNotificationToAllSubscribers(SendNotificationToAllSubscriberRequest request) {

        var sendNotificationResponse = new SendNotificationResponse();
        var headings = request.Messages.ToDictionary(m => m.Language, m => m.Title);
        var contents = request.Messages.ToDictionary(m => m.Language, m => m.Content);
        var oneSignalRequest = new OneSignalNotificationRequestBase()
        {
            Headings = headings,
            Contents = contents,
            AppId = "d4ea2d90-0856-4162-afc4-f488815a9357",
            Segments = request.TestMode
                ? new List<string>() { "All Test User" }
                : new List<string>() { "Active Users", "Subscribed Users", "Inactive Users", "Engaged Users" },
        };

        var apiBaseResponse = await OneSignalPushNotification(oneSignalRequest);

        if (apiBaseResponse.ErrorCode == ApiErrorEnum.NoError)
        {
            var data = apiBaseResponse.Result;
            sendNotificationResponse.Id = data.Id;
            sendNotificationResponse.Recipients = data.Recipients;
            sendNotificationResponse.SentAt = data.SentAt;
        }

        return new ApiBaseResponse<SendNotificationResponse>(sendNotificationResponse);
    }




    public async Task<ApiBaseResponse<SendNotificationResponse>> OneSignalPushNotification(OneSignalNotificationRequestBase oneSignalNotification)
    {

        var apiBaseResponse = new ApiBaseResponse<SendNotificationResponse>(
            await _http.SendNotification<SendNotificationResponse, OneSignalNotificationRequestBase>(
                "https://onesignal.com/api/v1/notifications", oneSignalNotification,
                EnumHttpContentType.Json,"YmM3Zjc4N2ItZTA4Mi00NGJmLThlZjUtMDUxYzhlMzdhMTRh"));
        apiBaseResponse.Result.SentAt = DateTime.Now;
        return apiBaseResponse;

    }
}

 