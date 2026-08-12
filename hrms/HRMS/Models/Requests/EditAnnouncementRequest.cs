using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class EditAnnouncementRequest : AddAnnouncementRequest
    {
        [JsonProperty("Id")]
        public int Id { get; set; }        
        
        [JsonIgnore]
        public int ModifiedBy { get; set; }

        public override void CheckModels()
        {
            if (Id <= 0)
            {
                ThrowInvalidModelException("Id is required");
            }
            base.CheckModels();
        }
    }
}
