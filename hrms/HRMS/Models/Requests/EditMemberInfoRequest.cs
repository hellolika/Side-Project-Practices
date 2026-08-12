using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class EditMemberInfoRequest : Member
    {
        [JsonIgnore]
        public override string Password { get; set; }

        [JsonProperty("IsResigned")]
        public bool IsResigned { get; set; }
        
        [JsonIgnore] public override string DepartmentName { get; set; }

        public void CheckModels()
        {
            if (MemberId <= 0)
            {
                ThrowInvalidModelException("Id is required");
            }
            CheckCommonModels();
        }
    }
}
