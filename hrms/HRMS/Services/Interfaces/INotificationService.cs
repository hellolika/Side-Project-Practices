using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;

namespace HRMS.Services.Interfaces;


public interface INotificationService
{
    Task<ApiBaseResponse<SendNotificationResponse>> SendNotificationByExternalIds(SendNotificationByExternalIdRequest request);
    Task<ApiBaseResponse<SendNotificationResponse>> SendNotificationToAllSubscribers(SendNotificationToAllSubscriberRequest request);
    Task<ApiBaseResponse<SendNotificationResponse>> OneSignalPushNotification(OneSignalNotificationRequestBase oneSignalNotification);
}