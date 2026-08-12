using Newtonsoft.Json;

namespace HRMS.Models.Responses
{

    public class GetAnnouncementResponse
    {
        [JsonProperty("CurrentPage")]
        public int CurrentPage { get; set; }

        [JsonProperty("TotalPages")]
        public int TotalPages { get; set; }

        [JsonProperty("ItemPerPage")]
        public int ItemPerPage { get; set; }

        [JsonProperty("Announcements")]
        public List<GetAnnouncementResponseItem> Announcements { get; set; }
    }

    public class GetAnnouncementResponseItem
    {
        [JsonProperty("Id")]
        public int Id { get; set; }

        [JsonProperty("Title")]
        public string Title { get; set; }

        [JsonProperty("Message")]
        public string Message { get; set; }

        [JsonProperty("CreatedBy")]
        public string CreatedBy { get; set; }

        [JsonProperty("CreatedOn")]
        public DateTimeOffset CreatedOn { get; set; }

        [JsonProperty("ModifiedBy")]
        public string ModifiedBy { get; set; }

        [JsonProperty("ModifiedOn")]
        public DateTimeOffset ModifiedOn { get; set; }
    }
}