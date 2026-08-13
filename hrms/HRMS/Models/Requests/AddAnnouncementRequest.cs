using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class AddAnnouncementRequest
    {
        [JsonProperty("Title")]
        public string Title { get; set; }

        [JsonProperty("Message")]
        public string Message { get; set; }

        [JsonIgnore]
        public int CreatedBy { get; set; }

        public virtual void CheckModels()
        {
            if (string.IsNullOrWhiteSpace(Title))
            {
                ThrowInvalidModelException("Title is required");
            }
            if (string.IsNullOrWhiteSpace(Message))
            {
                ThrowInvalidModelException("Message is required");
            }
            if (Title.Length > 200)
            {
                ThrowInvalidModelException("Maximun characters allowed for title is 200 only");
            }
            if (Message.Length > 2000)
            {
                ThrowInvalidModelException("Maximun characters allowed for message is 2000 only");
            }
        }

        protected static void ThrowInvalidModelException(string message)
        {
            throw new ApiException(ApiErrorEnum.InvalidModelState, message);
        }
    }
}