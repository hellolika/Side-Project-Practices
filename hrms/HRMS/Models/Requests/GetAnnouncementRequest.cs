using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class GetAnnouncementRequest : BaseRequest
    {
        [JsonProperty("Page")]
        public int Page { get; set; } = 1;

        [JsonProperty("ItemPerPage")]
        public int ItemPerPage { get; set; } = 10;

        public override void CheckModels()
        {
            if(Page < 1)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Invalid page number");
            }
            base.CheckModels();
        }
    }
}
