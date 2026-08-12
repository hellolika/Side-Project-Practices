using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;
using System.Text.RegularExpressions;

namespace HRMS.Models.Requests
{
    public class RegisterMemberRequest : Member
    {
        [JsonIgnore]
        public override int MemberId { get; set; }
        
        [JsonIgnore] public override string Password { get; set; } = "Techbodia123";
        [JsonIgnore] public override int Permission { get; set; } = 1;
        [JsonIgnore] public override int WorkLocationId { get; set; } = 1; 
        [JsonIgnore] public override bool IsSupervisor { get; set; } = false;
        
        [JsonIgnore] public override bool IsDeleted { get; set; } = false;
        
        [JsonIgnore] public override string DepartmentName { get; set; }

        public void CheckModels()
        {
            CheckCommonModels();
        }
    }
}