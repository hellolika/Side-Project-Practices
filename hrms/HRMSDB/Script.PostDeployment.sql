/*
Post-Deployment Script Template							
--------------------------------------------------------------------------------------
 This file contains SQL statements that will be appended to the build script.		
 Use SQLCMD syntax to include a file in the post-deployment script.			
 Example:      :r .\myfile.sql								
 Use SQLCMD syntax to reference a variable in the post-deployment script.		
 Example:      :setvar TableName MyTable							
               SELECT * FROM [$(TableName)]					
--------------------------------------------------------------------------------------
*/
IF ('$(Mode)' = 'SchemaOnly')
SET NOEXEC ON; 
    print 'FullRelease:: ==============Post Deployment For CobwebDB================'
    :r Data\Team.sql
    :r Data\LeaveType.sql
    :r Data\LocationDetail.sql
    :r Data\LeaveAmount.sql
    :r Data\TransferType.sql
    :r Data\RoleType.sql
    :r Data\PermissionCategory.sql
    :r Data\PermissionType.sql
    :r Data\PopulateSuperAdminPermission.sql
    :r Data\BeneficiaryType.sql
    :r Data\Position.sql
SET NOEXEC OFF;
